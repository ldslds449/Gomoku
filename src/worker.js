let gpu;
let fallback = "";
let mode = "auto";
function report(initial) {
  const available = Boolean(gpu) || mode !== "gpu";
  postMessage({
    type: "backend",
    initial,
    available,
    name: gpu ? "GPU（WebGPU）" : available ? "CPU（WASM）" : "GPU 無法使用",
    reason: fallback,
  });
}
// The AI's search, kept after its move. Pondering continues below the played
// move while the player thinks, and the next search continues from the
// player's reply. `board` is the position after the AI's move, `reused`
// counts the AI's own simulations below that move, and `simulations` counts
// pondering work.
let kept;
let ponderTimer;
let pendingRound;
let lastPonderReport = 0;
let ponderRequest;
function stopPonder() {
  clearTimeout(ponderTimer);
  ponderTimer = undefined;
  return pendingRound;
}
// GPU dispatches kept in flight while the AI thinks: one runs while the next
// batch is selected.
const GPU_DEPTH = 2;
// Finish every selected leaf on the CPU after a GPU error. Returns false when
// GPU mode cannot continue without the device.
function failGpu(search, batch, error) {
  search.cpu_pending(batch);
  gpu.free();
  gpu = null;
  fallback = String(error);
  if (mode === "gpu") {
    report(true);
    return false;
  }
  report(false);
  return true;
}
// With the leaf setting on automatic, the leaves per dispatch follow the
// measured rounds per second: double while that gains over 10%, and step back
// when half as many leaves come within 5%, since fewer pending leaves keep
// selection fresher. Other sizes are forgotten now and then to re-explore.
const tuner = { batch: 0, leaves: 16, rates: new Map(), windows: 0 };
function leavesFor(config, batch) {
  const cap = wasm_bindgen.Gpu.max_leaves(batch);
  if (config.leaves > 0) return Math.min(config.leaves, cap);
  if (tuner.batch !== batch) {
    tuner.batch = batch;
    tuner.leaves = Math.min(16, cap);
    tuner.rates.clear();
  }
  return tuner.leaves;
}
function observeRate(rate) {
  const leaves = tuner.leaves;
  const old = tuner.rates.get(leaves);
  tuner.rates.set(leaves, old === undefined ? rate : (old + rate) / 2);
  if (++tuner.windows % 16 === 0)
    for (const size of [...tuner.rates.keys()])
      if (size !== leaves) tuner.rates.delete(size);
  const current = tuner.rates.get(leaves);
  const fewer = tuner.rates.get(leaves / 2);
  const more = tuner.rates.get(leaves * 2);
  const cap = Math.min(256, wasm_bindgen.Gpu.max_leaves(tuner.batch));
  if (leaves / 2 >= 4 && fewer !== undefined && fewer >= current * 0.95)
    tuner.leaves = leaves / 2;
  else if (leaves * 2 <= cap && (more === undefined || more > current * 1.1))
    tuner.leaves = leaves * 2;
}
// Helper workers search the same position with other seeds; the root
// statistics are merged (root parallelization). Each helper keeps its own tree
// and ponders like the main worker. Only the main worker spawns helpers.
let helpers = [];
let helperKey;
let helperRequest = 0;
// In a helper: the finished search, waiting for the move actually played.
let held;
function setHelpers(config) {
  const wanted =
    self.isHelper || !self.Worker
      ? 0
      : Math.max(
          0,
          (config.threads > 0
            ? config.threads
            : // Automatic: CPU cores, but one GPU device until helpers
              // sharing a GPU are measured in browsers.
              gpu
              ? 1
              : Math.min(4, self.navigator?.hardwareConcurrency ?? 1)) - 1,
        );
  const key = JSON.stringify([wanted, mode]);
  if (key === helperKey) return;
  helperKey = key;
  for (const helper of helpers) helper.worker.terminate();
  helpers = [];
  for (let k = 0; k < wanted; k++) {
    const worker = new Worker(self.location.href);
    const helper = { worker, waiting: new Map() };
    helper.ready = new Promise((resolve) => {
      worker.onmessage = ({ data }) => {
        if (data.type === "backend") resolve();
        if (data.type === "helper-ponder") helper.ponder = data;
        if (data.type === "helper-stats") {
          helper.waiting.get(data.request)?.(data);
          helper.waiting.delete(data.request);
        }
      };
      worker.onerror = () => resolve();
    });
    worker.postMessage({ type: "helper" });
    worker.postMessage({ type: "connect", device: mode });
    helpers.push(helper);
  }
}
// The seed of helper k, distinct from the main search.
const helperConfig = (config, k) => ({
  ...config,
  seed: (config.seed ^ Math.imul(k + 1, 0x9e3779b9)) >>> 0,
});
function askHelpers(message, config) {
  for (const [k, helper] of helpers.entries()) {
    // Counts from an earlier pondering no longer describe this one.
    helper.ponder = undefined;
    helper.worker.postMessage({ ...message, config: helperConfig(config, k) });
  }
}
// Start the helpers' searches; resolves to their statistics (null on failure).
function helperSearches(board, root, config) {
  return helpers.map((helper, k) => {
    const request = ++helperRequest;
    const result = new Promise((resolve) => {
      helper.waiting.set(request, resolve);
      // A helper that never answers must not hold up the move.
      const limit = (config.mode === "time" ? config.ms : 60000) + 5000;
      setTimeout(() => resolve(null), limit);
    });
    helper.ready.then(() =>
      helper.worker.postMessage({
        board,
        root,
        config: helperConfig(config, k),
        request,
        helper: true,
      }),
    );
    return result;
  });
}
function clearKept() {
  kept?.search.free();
  kept = undefined;
}
function keep(search, board, root, point, config) {
  clearKept();
  if (!search.focus(point)) return false;
  const after = board.slice();
  after[point] = root;
  kept = {
    search,
    board: after,
    root,
    key: searchKey(config),
    batch: config.batch,
    reused: search.focus_visits(),
    simulations: 0,
    deadline: 0,
  };
  return true;
}
const searchKey = ({ seed, explore, rule, batch, policy, formula }) =>
  JSON.stringify([seed, explore, rule, batch, Boolean(policy), formula]);
// PUCT priors follow (candidate score + 1) to this power.
const PRIOR_POWER = 1.5;
function needsSearch(rounds, elapsed, config) {
  return config.mode === "rounds"
    ? rounds < config.rounds
    : rounds === 0 || elapsed < config.ms;
}
function createSearch(board, root, config) {
  const search = new wasm_bindgen.Search(
    board,
    root,
    config.seed,
    config.explore,
    config.rule,
  );
  search.set_policy(Boolean(config.policy));
  // PUCT unless UCT is chosen; the exploration constant serves either.
  if (config.formula !== "uct") search.set_puct(config.explore, PRIOR_POWER);
  return search;
}
function ponderStatus(state, reason) {
  let simulations = kept?.simulations ?? 0;
  let reused = kept?.reused ?? 0;
  // A helper reports its counts to the main worker, which adds them up.
  if (self.isHelper) {
    postMessage({ type: "helper-ponder", simulations, reused });
    return;
  }
  for (const helper of helpers) {
    simulations += helper.ponder?.simulations ?? 0;
    reused += helper.ponder?.reused ?? 0;
  }
  postMessage({
    type: "ponder-status",
    request: ponderRequest,
    state,
    reason,
    // Counted over the main tree and every helper's.
    simulations,
    reused,
    // The player's replies as searched so far, best first for the player.
    hint: kept
      ? wasm_bindgen.rank_stats(kept.search.focus_stats(), kept.batch, false)
      : [],
  });
}
async function ponderRound(entry, config) {
  const batch = config.batch;
  if (gpu) {
    // Like the foreground search, select the next batch while one runs.
    try {
      if (gpu.flights() < GPU_DEPTH) {
        const leaves = entry.search.select_batch(
          leavesFor(config, batch),
          batch,
        );
        gpu.submit(entry.search, batch);
        entry.simulations += leaves * batch;
      }
      if (gpu.flights() >= GPU_DEPTH) await gpu.finish(entry.search);
    } catch (error) {
      entry.search.cpu_pending(batch);
      gpu.free();
      gpu = null;
      fallback = String(error);
      report(false);
      if (mode === "gpu") {
        ponderTimer = undefined;
        ponderStatus("unavailable", "GPU 發生錯誤，請切換裝置");
      }
      // Otherwise the next background round uses the CPU.
    }
    return;
  }
  const terminal = entry.search.select();
  if (terminal >= 0) entry.search.update(terminal, 1);
  else {
    let tally = 0;
    const slice = config.rule === 2 ? 32 : 128;
    for (let begin = 0; begin < batch; begin += slice) {
      tally += entry.search.cpu_batch_slice(batch, begin, slice);
      if (begin + slice < batch) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (ponderTimer === undefined || performance.now() >= entry.deadline) {
          entry.search.cancel_selection();
          return;
        }
      }
    }
    entry.search.update_simulations(tally, batch);
    entry.simulations += batch;
  }
}
// Back up the pondering dispatches still on the GPU, so the kept tree has no
// leaves in flight before it is rerooted, replaced or freed.
async function drainPonder() {
  while (gpu && kept && gpu.flights() > 0) {
    try {
      await gpu.finish(kept.search);
    } catch (error) {
      kept.search.cpu_pending(kept.batch);
      throw error;
    }
  }
}
function queuePonder(config) {
  if (performance.now() >= kept.deadline || kept.search.solved() >= 0) {
    ponderTimer = undefined;
    pendingRound = drainPonder()
      .catch(() => {})
      .then(() => ponderStatus("done"));
    return;
  }
  ponderTimer = setTimeout(() => {
    pendingRound = (async () => {
      const entry = kept;
      if (entry && performance.now() < entry.deadline)
        await ponderRound(entry, config);
    })();
    const finished = pendingRound;
    finished
      .then(
        () => {
          // The foreground message clears the timer and takes ownership of this round.
          if (ponderTimer !== undefined && kept) {
            const now = performance.now();
            if (now - lastPonderReport >= 120) {
              ponderStatus("running");
              lastPonderReport = now;
            }
            queuePonder(config);
          }
        },
        () => {
          clearKept();
          ponderStatus("unavailable", "預先搜尋執行失敗");
        },
      )
      .finally(() => {
        if (pendingRound === finished) pendingRound = undefined;
      });
  }, 0);
}
onmessage = async ({ data }) => {
  const { board, root, config } = data;
  let search;
  let answered = false;
  try {
    await engineReady;
    if (data.type === "position-evaluation") {
      postMessage({
        type: "position-evaluation",
        request: data.request,
        winner: wasm_bindgen.forced_winner(board, data.player, data.rule),
      });
      return;
    }
    if (data.type === "check-black") {
      postMessage({
        type: "black-checked",
        legal: wasm_bindgen.black_can_move(board),
      });
      return;
    }
    if (data.type === "validate") {
      postMessage({
        type: "validated",
        point: data.point,
        legal: wasm_bindgen.legal_move(
          board,
          data.point,
          data.player,
          data.rule,
        ),
      });
      return;
    }
    if (data.type === "helper") {
      self.isHelper = true;
      return;
    }
    if (data.type === "helper-played") {
      if (
        held &&
        !keep(held.search, held.board, held.root, data.point, held.config)
      )
        held.search.free();
      held = undefined;
      return;
    }
    if (data.type === "connect") {
      mode = data.device;
      if (mode !== "cpu") {
        try {
          if (!self.navigator?.gpu) throw Error("瀏覽器未提供 WebGPU");
          const adapter = await self.navigator.gpu.requestAdapter();
          const info = adapter?.info;
          const width =
            info?.vendor === "intel" && info.architecture === "gen-9" ? 8 : 32;
          gpu = await wasm_bindgen.Gpu.open(width);
        } catch (error) {
          fallback = String(error);
        }
      }
      report(true);
      return;
    }
    const start = performance.now();
    // Only one GPU dispatch can own the mapped readback buffer at a time.
    await stopPonder()?.catch(() => {});
    await drainPonder().catch(() => {});
    if (
      !self.isHelper &&
      (data.type === "ponder" || data.type === "cancel-ponder")
    ) {
      if (config) setHelpers(config);
      askHelpers(data, config ?? {});
    }
    // Stopping keeps the tree: a later search may still continue from it.
    if (data.type === "cancel-ponder") return;
    if (data.type === "ponder") {
      ponderRequest = data.request;
      if (!board.some((cell) => cell === 1 || cell === 2)) return;
      if (!gpu && mode === "gpu") {
        ponderStatus("unavailable", "GPU 未就緒");
        return;
      }
      const matches =
        kept?.key === searchKey(config) &&
        kept.root === root &&
        kept.board.every((value, i) => value === board[i]);
      if (!matches) {
        // Without the AI's tree (a restored game, undo or new settings),
        // rebuild it from the position before the AI's last move.
        clearKept();
        const last = data.last;
        if (Number.isInteger(last) && board[last] === root) {
          const before = board.slice();
          before[last] = 0;
          try {
            const tree = createSearch(before, root, config);
            if (!keep(tree, before, root, last, config)) tree.free();
          } catch {
            // The position before the move was not a valid search root.
          }
        }
        if (!kept) {
          ponderStatus("unavailable", "無法建立搜尋樹");
          return;
        }
      }
      kept.batch = config.batch;
      kept.deadline =
        config.ponderSeconds === 0
          ? Infinity
          : start + 1000 * (config.ponderSeconds ?? 60);
      ponderStatus("running");
      queuePonder(config);
      return;
    }
    if (mode === "gpu" && !gpu) {
      clearKept();
      report(true);
      return;
    }
    // A helper's unclaimed search from an earlier move is no longer needed.
    held?.search.free();
    held = undefined;
    let gpuMs = 0;
    // Continue the kept tree when the board is its position plus a reply.
    const cached =
      kept?.key === searchKey(config) &&
      kept.root === root &&
      kept.search.reroot(board)
        ? kept
        : undefined;
    if (cached) kept = undefined;
    else clearKept();
    search = cached?.search ?? createSearch(board, root, config);
    // The budget counts only this move's work; the inherited tree is extra.
    let rounds = 0;
    // Both count the helpers' trees too once their results are merged.
    let reusedSimulations = search.visits();
    let simulations = reusedSimulations;
    const batchSize = config.batch ?? 64;
    if (![64, 128, 256, 512, 1024, 2048, 4096, 8192].includes(batchSize))
      throw Error("無效的批次大小");
    const reply = (i, reason, children, backend, extra = {}) => {
      const [total, wins, draws] =
        children.find((child) => child.i === i)?.outcomes ??
        search.move_outcomes(i);
      const depth = Math.max(search.depth(), extra.depth ?? 0);
      // Keep the tree for pondering and the next move.
      if (keep(search, board, root, i, config)) search = undefined;
      postMessage({
        i,
        reason,
        children,
        backend,
        config,
        rounds,
        simulations,
        reusedSimulations,
        selected: { total, wins, draws },
        gpuMs,
        fallback,
        depth,
        // Proof of the chosen move for the AI: "win", "loss", "draw" or "".
        proven:
          extra.proven ?? children.find((child) => child.i === i)?.proven ?? "",
        elapsed: performance.now() - start,
        threads: helpers.length + 1,
        leaves: extra.leaves ?? 0,
      });
    };
    if (!search.has_moves()) {
      postMessage({ type: "no-moves" });
      return;
    }
    const direct = search.direct();
    if (direct >= 0 && data.helper) {
      answered = true;
      postMessage({ type: "helper-stats", request: data.request, stats: [] });
      return;
    }
    // A forced move still reports a result proven earlier in the reused tree.
    const solved = ["loss", "draw", "win"][search.solved()];
    if (direct >= 0)
      return reply(
        direct,
        board.some((v) => v === 1 || v === 2) ? "戰術或唯一候選" : "天元開局",
        [],
        "CPU（戰術判定）",
        solved ? { proven: solved } : {},
      );
    // A VCF or VCT found before searching is a proven win: play its first move.
    const vcf = search.vcf_move();
    if (vcf >= 0 && data.helper) {
      answered = true;
      postMessage({ type: "helper-stats", request: data.request, stats: [] });
      return;
    }
    if (vcf >= 0)
      return reply(
        vcf,
        `${search.by_vct() ? "VCT" : "VCF"} 必勝（${search.vcf_plies()} 手內連五）`,
        [],
        search.by_vct() ? "CPU（VCT）" : "CPU（VCF）",
        { proven: "win" },
      );
    const foregroundStart = performance.now();
    let backend = gpu ? "Rust wgpu" : "Rust CPU";
    if (!data.helper) setHelpers(config);
    const helperResults = data.helper
      ? []
      : helperSearches(board, root, config);
    // Rounds per second over windows of dispatches, for the leaf tuner.
    let windowStart = performance.now();
    let windowRounds = 0;
    let windowFinishes = 0;
    let leavesUsed = 0;
    // Stop early once the solver proves the root position.
    while (
      search.solved() < 0 &&
      needsSearch(rounds, performance.now() - foregroundStart, config)
    ) {
      if (gpu) {
        // One dispatch scores many leaves selected with virtual loss. The next
        // batch is selected while the previous one runs on the GPU.
        const before = search.rounds();
        const room =
          config.mode === "rounds"
            ? config.rounds - rounds - search.in_flight()
            : Infinity;
        leavesUsed = leavesFor(config, batchSize);
        const limit = Math.min(leavesUsed, room);
        try {
          if (limit > 0 && gpu.flights() < GPU_DEPTH) {
            const leaves = search.select_batch(limit, batchSize);
            gpu.submit(search, batchSize);
            simulations += leaves * batchSize;
          }
          if (gpu.flights() >= GPU_DEPTH || (limit <= 0 && gpu.flights() > 0)) {
            const wait = performance.now();
            await gpu.finish(search);
            gpuMs += performance.now() - wait;
            windowFinishes++;
          }
        } catch (error) {
          if (!failGpu(search, batchSize, error)) return;
          backend = "Rust CPU";
        }
        rounds += search.rounds() - before;
        windowRounds += search.rounds() - before;
        const span = performance.now() - windowStart;
        if (!(config.leaves > 0) && windowFinishes >= 4 && span >= 25) {
          observeRate(windowRounds / span);
          windowStart = performance.now();
          windowRounds = 0;
          windowFinishes = 0;
        }
        continue;
      }
      const terminal = search.select();
      if (terminal >= 0) search.update(terminal, 1);
      else {
        search.update_simulations(search.cpu_batch(batchSize), batchSize);
        simulations += batchSize;
      }
      rounds++;
    }
    // Time mode may overshoot the deadline by the dispatches in flight.
    while (gpu?.flights() > 0) {
      const before = search.rounds();
      try {
        await gpu.finish(search);
      } catch (error) {
        if (!failGpu(search, batchSize, error)) return;
        backend = "Rust CPU";
      }
      rounds += search.rounds() - before;
    }
    if (data.helper) {
      // Wait for the main worker's choice before keeping the tree.
      held = { search, board, root, config };
      answered = true;
      postMessage({
        type: "helper-stats",
        request: data.request,
        stats: search.stats(),
        rounds,
        simulations,
        reusedSimulations,
        depth: search.depth(),
      });
      search = undefined;
      return;
    }
    // Merge the helpers' root statistics into this search's.
    let values = search.stats();
    let helperDepth = 0;
    for (const result of await Promise.all(helperResults)) {
      if (!result?.stats?.length) continue;
      values = wasm_bindgen.merge_stats(values, result.stats);
      rounds += result.rounds;
      simulations += result.simulations;
      reusedSimulations += result.reusedSimulations ?? 0;
      helperDepth = Math.max(helperDepth, result.depth);
    }
    const children = [];
    // Proof codes: 0 unknown, 1 root loss, 2 draw, 3 root win. `plies` counts
    // moves to the end of a proven game after this one. Listed in the order
    // the AI ranks them.
    const ranked = wasm_bindgen.rank_stats(values, batchSize, true);
    for (let j = 0; j < ranked.length; j += 8)
      children.push({
        i: ranked[j],
        n: ranked[j + 1],
        w: ranked[j + 2] / 2,
        proven: ["", "loss", "draw", "win"][ranked[j + 3]],
        plies: ranked[j + 4],
        outcomes: [ranked[j + 5], ranked[j + 6], ranked[j + 7]],
      });
    // Difficulty samples among searched moves; temperature 0 plays the best move.
    const point = wasm_bindgen.pick_move(
      values,
      config.temperature ?? 0,
      Math.random(),
      batchSize,
    );
    for (const helper of helpers)
      helper.worker.postMessage({ type: "helper-played", point });
    const proof = children.find((child) => child.i === point)?.proven;
    reply(
      point,
      { win: "MCTS（已證明必勝）", loss: "MCTS（已證明必敗）" }[proof] ??
        "MCTS",
      children,
      backend,
      { depth: helperDepth, leaves: leavesUsed },
    );
  } catch (error) {
    postMessage({ error: "AI 執行失敗：" + String(error) });
  } finally {
    search?.free();
    // A helper always answers a search, empty when it could not search.
    if (data.helper && !answered)
      postMessage({ type: "helper-stats", request: data.request, stats: [] });
  }
};
