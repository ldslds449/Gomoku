use super::Search;
use std::cell::RefCell;
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;
use wgpu::*;
// One set of buffers for a dispatch in flight.
struct Slot {
    input: Buffer,
    output: Buffer,
    read: Buffer,
    group: BindGroup,
}
// A submitted dispatch: its slot, result size, and the map callback's signal.
struct Flight {
    slot: usize,
    size: u64,
    batch: u32,
    leaves: u32,
    // Native polling waits for this submission; WebGPU maps on its own.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    index: SubmissionIndex,
    mapped: futures_channel::oneshot::Receiver<Result<(), BufferAsyncError>>,
}
#[wasm_bindgen]
pub struct Gpu {
    width: u32,
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
    slots: Vec<Slot>,
    // Oldest first; results come back in submission order.
    flights: RefCell<VecDeque<Flight>>,
}
// One dispatch scores up to 256 leaves and 65536 games.
const MAX_LEAVES: u32 = 256;
const MAX_GAMES: u32 = 65536;
// Dispatches that may be in flight at once.
const SLOTS: usize = 3;
const LEAF_BYTES: u64 = 644;
fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}
#[wasm_bindgen]
impl Gpu {
    pub async fn open(width: Option<u32>) -> Result<Gpu, JsError> {
        // 32 measured slightly faster than 64 on Apple GPUs; wider groups lose.
        let width = width.unwrap_or(32);
        if ![8, 32, 64].contains(&width) {
            return Err(JsError::new("不支援的工作群組大小"));
        }
        Self::with_width(width).await
    }
}
impl Gpu {
    async fn with_width(width: u32) -> Result<Gpu, JsError> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .map_err(err)?;
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .map_err(err)?;
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("rollout.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions {
                constants: &[("width", f64::from(width))],
                ..Default::default()
            },
            cache: None,
        });
        let buf = |size, usage| {
            device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let max_result_bytes = u64::from(MAX_GAMES / width) * 4;
        let slots = (0..SLOTS)
            .map(|_| {
                let input = buf(
                    LEAF_BYTES * u64::from(MAX_LEAVES),
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                );
                let output = buf(
                    max_result_bytes,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                );
                let read = buf(
                    max_result_bytes,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                );
                let group = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: input.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: output.as_entire_binding(),
                        },
                    ],
                });
                Slot {
                    input,
                    output,
                    read,
                    group,
                }
            })
            .collect();
        Ok(Self {
            width,
            device,
            queue,
            pipeline,
            slots,
            flights: Default::default(),
        })
    }
}
fn check_batch(batch: u32) -> Result<(), JsError> {
    if !(64..=8192).contains(&batch) || !batch.is_multiple_of(64) {
        return Err(JsError::new("無效的批次大小"));
    }
    Ok(())
}
#[wasm_bindgen]
impl Gpu {
    // Leaves one dispatch can score at this batch size.
    pub fn max_leaves(batch: u32) -> u32 {
        (MAX_GAMES / batch.max(1)).clamp(1, MAX_LEAVES)
    }
    // Dispatches that may be in flight at once.
    pub fn slots() -> u32 {
        SLOTS as u32
    }
    // Dispatches submitted and not yet finished.
    pub fn flights(&self) -> u32 {
        self.flights.borrow().len() as u32
    }
    // Start scoring the pending leaves without waiting. Their virtual losses
    // stay on the tree, so the next batch can be selected meanwhile; `finish`
    // backs up the oldest dispatch. On error the leaves stay with the search
    // for `Search::cpu_pending`.
    pub fn submit(&self, s: &mut Search, batch: u32) -> Result<u32, JsError> {
        check_batch(batch)?;
        let leaves = s.pending.len() as u32;
        if leaves == 0 {
            return Ok(0);
        }
        if leaves > Self::max_leaves(batch) {
            return Err(JsError::new("待模擬的葉節點過多"));
        }
        let mut bytes = Vec::with_capacity((LEAF_BYTES * u64::from(leaves)) as usize);
        for p in &s.pending {
            Self::encode_leaf(&mut bytes, s, &p.leaf, p.turn, p.round.wrapping_mul(batch));
        }
        self.launch(&bytes, batch, leaves)?;
        s.queue_pending();
        Ok(leaves)
    }
    // Wait for the oldest dispatch and back up its leaves in selection order.
    pub async fn finish(&self, s: &mut Search) -> Result<u32, JsError> {
        let Some((batch, leaves, tallies)) = self.land_sums().await? else {
            return Ok(0);
        };
        s.land(&tallies, batch);
        Ok(leaves)
    }
    // Score every pending leaf in one dispatch and wait for it.
    pub async fn run_pending(&self, s: &mut Search, batch: u32) -> Result<u32, JsError> {
        if self.flights() > 0 {
            return Err(JsError::new("仍有未完成的派送"));
        }
        let leaves = self.submit(s, batch)?;
        self.finish(s).await?;
        Ok(leaves)
    }
    pub async fn batch(&self, s: &Search, batch: u32) -> Result<u32, JsError> {
        self.read_scores(s, batch, |words| {
            words.map(|word| u32::from_le_bytes(*word)).sum()
        })
        .await
    }
}
impl Gpu {
    // Per-leaf sums of the oldest flight's workgroup totals.
    async fn land_sums(&self) -> Result<Option<(u32, u32, Vec<u32>)>, JsError> {
        let Some((batch, leaves, words)) = self.land().await? else {
            return Ok(None);
        };
        let groups = (batch / self.width) as usize;
        let mut sums = vec![0; leaves as usize];
        for (i, w) in words.into_iter().enumerate() {
            sums[i / groups] += w;
        }
        Ok(Some((batch, leaves, sums)))
    }
    async fn read_scores<T>(
        &self,
        s: &Search,
        batch: u32,
        reduce: impl FnOnce(std::slice::Iter<'_, [u8; 4]>) -> T,
    ) -> Result<T, JsError> {
        check_batch(batch)?;
        if self.flights() > 0 {
            return Err(JsError::new("仍有未完成的派送"));
        }
        let mut bytes = Vec::with_capacity(LEAF_BYTES as usize);
        Self::encode_leaf(&mut bytes, s, &s.leaf, s.turn, s.rounds.wrapping_mul(batch));
        self.launch(&bytes, batch, 1)?;
        let (_, _, words) = self.land().await?.unwrap();
        let bytes: Vec<[u8; 4]> = words.into_iter().map(u32::to_le_bytes).collect();
        Ok(reduce(bytes.iter()))
    }
    // Root, seed, rule and policy come from the search; the leaf from a selection.
    fn encode_leaf(bytes: &mut Vec<u8>, s: &Search, board: &super::Board, turn: u8, offset: u32) {
        let (root, seed, rule, policy) = (s.root, s.seed, s.rule, s.policy);
        let mut empty = [0u32; 57];
        let mut lines = [0u32; 88];
        let mut occupied = [0u32; 8];
        let mut count = 0;
        for (i, &player) in board.iter().enumerate() {
            if player == 0 {
                empty[count / 4] |= (i as u32) << ((count % 4) * 8);
                count += 1;
                continue;
            }
            occupied[i / 32] |= 1 << (i % 32);
            if player > 2 {
                continue;
            }
            let x = i % 15;
            let y = i / 15;
            let shift = (player - 1) as usize * 16;
            for (line, pos) in [(y, x), (15 + x, y), (44 + x - y, x), (59 + x + y, x)] {
                lines[line] |= 1 << (pos + shift);
            }
        }
        let threats = if policy {
            super::rollout_threats(board, &super::lines::Lines::from_board(board), rule)
        } else {
            [[super::NO_THREAT; 2]; 2]
        };
        let threats =
            u32::from_le_bytes([threats[0][0], threats[0][1], threats[1][0], threats[1][1]]);
        for word in empty
            .into_iter()
            .chain(lines)
            .chain([
                count as u32,
                turn as u32,
                root as u32,
                seed,
                offset,
                rule as u32,
            ])
            .chain(occupied)
            .chain([threats, u32::from(policy)])
        {
            bytes.extend(word.to_le_bytes())
        }
    }
    // Upload leaves into a free slot and submit their dispatch.
    fn launch(&self, bytes: &[u8], batch: u32, leaves: u32) -> Result<(), JsError> {
        let mut flights = self.flights.borrow_mut();
        if flights.len() >= SLOTS {
            return Err(JsError::new("派送佇列已滿"));
        }
        let slot = flights.back().map_or(0, |f| (f.slot + 1) % SLOTS);
        let buffers = &self.slots[slot];
        let size = u64::from(batch / self.width * leaves) * 4;
        self.queue.write_buffer(&buffers.input, 0, bytes);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.group, &[]);
            pass.dispatch_workgroups(batch / self.width, leaves, 1);
        }
        enc.copy_buffer_to_buffer(&buffers.output, 0, &buffers.read, 0, size);
        let index = self.queue.submit([enc.finish()]);
        let (tx, rx) = futures_channel::oneshot::channel();
        buffers
            .read
            .slice(..size)
            .map_async(MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        flights.push_back(Flight {
            slot,
            size,
            batch,
            leaves,
            index,
            mapped: rx,
        });
        Ok(())
    }
    // Wait for the oldest flight and read its workgroup totals.
    async fn land(&self) -> Result<Option<(u32, u32, Vec<u32>)>, JsError> {
        let Some(flight) = self.flights.borrow_mut().pop_front() else {
            return Ok(None);
        };
        // Natively, wait for this submission only, so later ones keep running.
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(PollType::Wait {
                submission_index: Some(flight.index.clone()),
                timeout: None,
            })
            .map_err(err)?;
        flight.mapped.await.map_err(err)?.map_err(err)?;
        let read = &self.slots[flight.slot].read;
        let words = read
            .slice(..flight.size)
            .get_mapped_range()
            .map_err(err)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        read.unmap();
        Ok(Some((flight.batch, flight.leaves, words)))
    }
}

#[cfg(all(
    test,
    any(target_os = "macos", target_os = "windows", target_os = "linux")
))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a Metal, Vulkan or DX12 GPU"]
    fn pending_dispatch_matches_cpu_scoring() {
        pollster::block_on(async {
            let gpu = Gpu::with_width(8).await.unwrap();
            let mut board = [0; 225];
            board[112] = 1;
            for batch in [64, 1024] {
                let leaves = Gpu::max_leaves(batch);
                let mut on_gpu = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                let mut on_cpu = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                // The large batch also checks the win-and-block policy.
                on_gpu.policy = batch > 64;
                on_cpu.policy = batch > 64;
                let mut times = [std::time::Duration::ZERO; 2];
                for _ in 0..6 {
                    let start = std::time::Instant::now();
                    let count = on_gpu.select_batch(leaves, batch);
                    gpu.run_pending(&mut on_gpu, batch).await.unwrap();
                    times[0] += start.elapsed();
                    let start = std::time::Instant::now();
                    assert_eq!(on_cpu.select_batch(leaves, batch), count);
                    on_cpu.cpu_pending(batch);
                    times[1] += start.elapsed();
                    assert!(on_gpu.pending.is_empty());
                }
                assert_eq!(on_gpu.stats(), on_cpu.stats());
                assert_eq!(on_gpu.rounds, on_cpu.rounds);
                assert!(on_gpu.depth() >= 2);
                println!(
                    "batch {batch}, {leaves} leaves x 6 dispatches: gpu {:?}, cpu {:?}",
                    times[0], times[1]
                );
            }
        })
    }
    #[test]
    #[ignore = "requires a Metal, Vulkan or DX12 GPU"]
    fn pending_throughput_benchmark() {
        pollster::block_on(async {
            let gpu = Gpu::with_width(64).await.unwrap();
            // A middle-game position from the five-free two-colour pattern.
            let mut board = [0; 225];
            let mut remaining = [21, 20];
            for step in 0..225 {
                let i = (step * 97) % 225;
                let side = ((i % 15 + 2 * (i / 15)) % 4) / 2;
                if remaining[side] > 0 {
                    board[i] = side as u8 + 1;
                    remaining[side] -= 1;
                }
            }
            for (batch, leaves) in [
                (64, 1),
                (64, 16),
                (64, 64),
                (64, 256),
                (256, 64),
                (256, 256),
                (1024, 16),
                (1024, 64),
            ] {
                let mut s = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                assert!(s.direct() < 0);
                let (mut select, mut dispatch) = (vec![], vec![]);
                for k in 0..25 {
                    let t = std::time::Instant::now();
                    assert_eq!(s.select_batch(leaves, batch), leaves);
                    let selected = t.elapsed();
                    let t = std::time::Instant::now();
                    gpu.run_pending(&mut s, batch).await.unwrap();
                    if k >= 5 {
                        select.push(selected);
                        dispatch.push(t.elapsed());
                    }
                }
                select.sort();
                dispatch.sort();
                let games = f64::from(batch * leaves);
                // The same work with selection overlapping two dispatches in flight.
                let mut s = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                let start = std::time::Instant::now();
                for _ in 0..20 {
                    s.select_batch(leaves, batch);
                    gpu.submit(&mut s, batch).unwrap();
                    if gpu.flights() >= 2 {
                        gpu.finish(&mut s).await.unwrap();
                    }
                }
                while gpu.flights() > 0 {
                    gpu.finish(&mut s).await.unwrap();
                }
                let sequential = 20. * (select[10] + dispatch[10]).as_secs_f64();
                let pipelined = start.elapsed().as_secs_f64();
                println!(
                    "batch {batch:4} x {leaves:3} leaves = {games:6} games: select {:>9.3?}, dispatch {:>9.3?}, {:5.2} M games/s on GPU; per batch sequential {:>7.3} ms, pipelined {:>7.3} ms",
                    select[10],
                    dispatch[10],
                    games / dispatch[10].as_secs_f64() / 1e6,
                    sequential / 20. * 1e3,
                    pipelined / 20. * 1e3
                );
            }
        })
    }
    // Games per second for one full dispatch (256 leaves x 256 games) from
    // positions of the five-free pattern, by workgroup width and rollout policy.
    #[test]
    #[ignore = "requires a Metal, Vulkan or DX12 GPU"]
    fn shader_throughput_benchmark() {
        pollster::block_on(async {
            for width in [32, 64, 128, 256] {
                let gpu = Gpu::with_width(width).await.unwrap();
                for pairs in [0, 10, 20] {
                    let mut board = [0; 225];
                    let mut remaining = [pairs + 1, pairs];
                    for step in 0..225 {
                        let i = (step * 97) % 225;
                        let side = ((i % 15 + 2 * (i / 15)) % 4) / 2;
                        if remaining[side] > 0 {
                            board[i] = side as u8 + 1;
                            remaining[side] -= 1;
                        }
                    }
                    for policy in [false, true] {
                        let mut s = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                        s.policy = policy;
                        let mut times = vec![];
                        for _ in 0..12 {
                            assert_eq!(s.select_batch(256, 256), 256);
                            let t = std::time::Instant::now();
                            gpu.run_pending(&mut s, 256).await.unwrap();
                            times.push(t.elapsed());
                        }
                        times.sort();
                        println!(
                            "width {width:3}, {:3} stones, policy {policy:5}: {:5.2} M games/s",
                            2 * pairs + 1,
                            65536. / times[6].as_secs_f64() / 1e6
                        );
                    }
                }
            }
        })
    }
    #[test]
    #[ignore = "requires a Metal, Vulkan or DX12 GPU"]
    fn pipelined_dispatches_match_cpu_scoring() {
        pollster::block_on(async {
            let gpu = Gpu::with_width(64).await.unwrap();
            let mut board = [0; 225];
            board[112] = 1;
            for (policy, depth) in [(false, 2), (true, 3)] {
                let mut on_gpu = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                let mut on_cpu = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                on_gpu.policy = policy;
                on_cpu.policy = policy;
                // Selection runs while earlier batches are still on the GPU.
                for _ in 0..10 {
                    let count = on_gpu.select_batch(16, 128);
                    assert_eq!(on_cpu.select_batch(16, 128), count);
                    gpu.submit(&mut on_gpu, 128).unwrap();
                    on_cpu.queue_pending();
                    if gpu.flights() >= depth {
                        gpu.finish(&mut on_gpu).await.unwrap();
                        on_cpu.cpu_finish(128);
                    }
                }
                while gpu.flights() > 0 {
                    gpu.finish(&mut on_gpu).await.unwrap();
                    on_cpu.cpu_finish(128);
                }
                assert_eq!(on_gpu.in_flight(), 0);
                assert_eq!(on_gpu.stats(), on_cpu.stats());
                assert_eq!(on_gpu.rounds, on_cpu.rounds);
            }
        })
    }
    #[test]
    #[ignore = "requires a Metal, Vulkan or DX12 GPU"]
    fn rollout_scores_match_individually() {
        pollster::block_on(async {
            let gpu = Gpu::with_width(1).await.unwrap();
            // The last fixtures start from black's open four against a white three,
            // so the leaf already holds winning points.
            for (inset, rule, fours) in [
                (0, 0, false),
                (1, 1, false),
                (3, 1, false),
                (0, 0, true),
                (0, 1, true),
            ] {
                let mut board = [0; 225];
                for (i, point) in board.iter_mut().enumerate() {
                    if !(inset..15 - inset).contains(&(i / 15))
                        || !(inset..15 - inset).contains(&(i % 15))
                    {
                        *point = 3;
                    }
                }
                board[112] = 1;
                if fours {
                    for i in [96, 97, 98, 99] {
                        board[i] = 1;
                    }
                    for i in [126, 127, 128, 200] {
                        board[i] = 2;
                    }
                }
                let mut s = Search::new(&board, 2, 0, 2f64.sqrt(), Some(rule)).unwrap();
                assert_eq!(s.select(), -1);
                let lines = crate::lines::Lines::from_board(&s.leaf);
                if fours {
                    assert_ne!(
                        super::super::rollout_threats(&s.leaf, &lines, rule)[0][0],
                        255
                    );
                }
                for (seed, policy) in [0, 42, 0xf2b58529, u32::MAX]
                    .into_iter()
                    .flat_map(|seed| [(seed, false), (seed, true)])
                {
                    s.seed = seed;
                    s.policy = policy;
                    let scores = gpu
                        .read_scores(&s, 256, |words| {
                            words.map(|w| u32::from_le_bytes(*w)).collect::<Vec<_>>()
                        })
                        .await
                        .unwrap();
                    for (id, score) in scores.into_iter().enumerate() {
                        assert_eq!(
                            score,
                            {
                                let (empty, n) = s.rollout_empty_points();
                                let points = s.rollout(
                                    lines.clone(),
                                    empty,
                                    n,
                                    seed ^ id as u32,
                                    s.rollout_threats(&lines),
                                );
                                points + (u32::from(points == 2) << 16)
                            },
                            "seed {seed}, policy {policy}, simulation {id}"
                        );
                    }
                }
            }
        });
    }
    #[test]
    #[ignore = "requires a Metal, Vulkan or DX12 GPU"]
    fn rollout_benchmark() {
        pollster::block_on(async {
            for width in [64, 32, 16, 8, 4, 1] {
                let gpu = Gpu::with_width(width).await.unwrap();
                for pairs in [0, 40, 92] {
                    // A periodic two-colour pattern keeps every fixture free of five-in-a-row.
                    let mut board = [0; 225];
                    let mut remaining = [pairs + 1, pairs];
                    for step in 0..225 {
                        let i = (step * 97) % 225;
                        let side = ((i % 15 + 2 * (i / 15)) % 4) / 2;
                        if remaining[side] > 0 {
                            board[i] = side as u8 + 1;
                            remaining[side] -= 1;
                        }
                    }
                    let mut search = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
                    assert_eq!(search.select(), -1);
                    for batch in [64, 256, 1024, 2048, 4096, 8192] {
                        let expected = search.cpu_batch(batch);
                        assert_eq!(gpu.batch(&search, batch).await.unwrap(), expected);
                        let mut times = Vec::new();
                        for _ in 0..30 {
                            let start = std::time::Instant::now();
                            assert_eq!(gpu.batch(&search, batch).await.unwrap(), expected);
                            times.push(start.elapsed());
                        }
                        times.sort();
                        println!(
                            "{} stones, width {width}, {batch} simulations: median {:?}, p95 {:?}",
                            pairs * 2 + 1,
                            times[15],
                            times[28]
                        );
                    }
                }
            }
        })
    }
}
