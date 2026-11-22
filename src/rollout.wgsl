struct Input {
  empty: array<u32, 57>,
  rows: array<u32, 15>,
  cols: array<u32, 15>,
  descending: array<u32, 29>,
  ascending: array<u32, 29>,
  count: u32,
  turn: u32,
  root: u32,
  seed: u32,
  offset: u32,
  rule: u32,
  // One bit per occupied or blocked point.
  occupied: array<u32, 8>,
  // Remembered winning points: byte 2 * (player - 1) + slot, 255 for none.
  threats: u32,
  // Nonzero to finish and block fives instead of playing only at random.
  policy: u32,
}

@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<storage, read_write> scores: array<u32>;

fn winning(value: u32, shift: u32, pos: u32, rule: u32) -> bool {
  let mask = (value >> shift) & 32767u;
  if (rule == 1u) {
    return
      countTrailingZeros(~(mask >> pos)) + countLeadingZeros(
        ~(mask << (31u - pos)),
      ) == 6u;
  }
  let window = (mask << 4u) >> pos;
  // For each right-hand pattern, select the minimum winning left nibble.
  let right = (window >> 5u) & 15u;
  let thresholds = select(0x8fefcfefu, 0x0fefcfefu, right >= 8u);
  return (window & 15u) >= extractBits(thresholds, (right & 7u) * 4u, 4u);
}

// Whether a stone of `q` at the empty (x, y) completes five on any axis.
fn wins_at(
  row: u32,
  col: u32,
  down: u32,
  up: u32,
  x: u32,
  y: u32,
  q: u32,
  rule: u32,
) -> bool {
  let shift = (q - 1u) * 16u;
  let bx = 1u << (x + shift);
  return
    winning(row | bx, shift, x, rule)
      || winning(col | (1u << (y + shift)), shift, y, rule)
      || winning(down | bx, shift, x, rule)
      || winning(up | bx, shift, x, rule);
}

// At least four of the player's stones within four cells of `pos` on the line.
fn crowded(value: u32, shift: u32, pos: u32) -> bool {
  let own = (value >> shift) & 32767u;
  return countOneBits(((own << 4u) >> pos) & 511u) >= 4u;
}

override width: u32 = 64u;

var<workgroup> total: atomic<u32>;

@compute @workgroup_size(width)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3u,
  @builtin(num_workgroups) grid: vec3u,
) {
  let inp = &inputs[group.y];
  let id = gid.x;
  var rows = inp.rows;
  var cols = inp.cols;
  var descending = inp.descending;
  var ascending = inp.ascending;
  var empty = inp.empty;
  var n = inp.count;
  var s = inp.seed ^ (inp.offset + id);
  if (s == 0u) {
    s = 1u;
  }
  var occupied = inp.occupied;
  var threats = inp.threats;
  var p = inp.turn;
  var score = 1u;
  while (n > 0u) {
    // Finish an own five, else block the opponent's, else play at random.
    var forced = 255u;
    for (var k = 0u; k < 4u && forced == 255u && inp.policy != 0u; k++) {
      let q = select(3u - p, p, k < 2u);
      let c = extractBits(threats, ((q - 1u) * 2u + (k & 1u)) * 8u, 8u);
      if (c == 255u || ((occupied[c >> 5u] >> (c & 31u)) & 1u) != 0u) {
        continue;
      }
      // An empty winning point stays winning unless exact five rules let a
      // longer line spoil it.
      let cx = c % 15u;
      let cy = c / 15u;
      if
        (inp.rule != 1u || wins_at(
          rows[cy],
          cols[cx],
          descending[14u + cx - cy],
          ascending[cx + cy],
          cx,
          cy,
          q,
          inp.rule,
        ))
      {
        forced = c;
      }
    }
    var j = 0u;
    if (forced != 255u) {
      while (extractBits(empty[j >> 2u], (j & 3u) * 8u, 8u) != forced) {
        j++;
      }
    } else {
      loop {
        s ^= s << 13u;
        s ^= s >> 17u;
        s ^= s << 5u;
        if (s > (0xffffffffu % n)) {
          break;
        }
      }
      j = (s - 1u) % n;
    }
    let word = empty[j >> 2u];
    let shift_j = (j & 3u) * 8u;
    let i = extractBits(word, shift_j, 8u);
    n--;
    let replacement = extractBits(empty[n >> 2u], (n & 3u) * 8u, 8u);
    empty[j >> 2u] = insertBits(word, replacement, shift_j, 8u);
    let x = i % 15u;
    let y = i / 15u;
    let shift = (p - 1u) * 16u;
    let bit = 1u << (x + shift);
    let down = 14u + x - y;
    let up = x + y;
    rows[y] |= bit;
    cols[x] |= 1u << (y + shift);
    descending[down] |= bit;
    ascending[up] |= bit;
    occupied[i >> 5u] |= 1u << (i & 31u);
    if
      (winning(rows[y], shift, x, inp.rule)
        || winning(cols[x], shift, y, inp.rule)
        || winning(descending[down], shift, x, inp.rule)
        || winning(ascending[up], shift, x, inp.rule))
    {
      score = select(0u, 65538u, p == inp.root);
      break;
    }
    // New winning points lie within four cells on a line crowded by the move,
    // and only that line can gain a five through them.
    let lines =
      array<u32, 4>(rows[y], cols[x], descending[down], ascending[up]);
    let positions = array<u32, 4>(x, y, x, x);
    let dxs = array<i32, 4>(1, 0, 1, 1);
    let dys = array<i32, 4>(0, 1, 1, -1);
    for (var a = 0u; a < 4u && inp.policy != 0u; a++) {
      if (!crowded(lines[a], shift, positions[a])) {
        continue;
      }
      for (var d = -4; d <= 4; d++) {
        let cx = i32(x) + dxs[a] * d;
        let cy = i32(y) + dys[a] * d;
        if (d == 0 || cx < 0 || cx > 14 || cy < 0 || cy > 14) {
          continue;
        }
        let ux = u32(cx);
        let uy = u32(cy);
        let c = uy * 15u + ux;
        let cpos = select(ux, uy, a == 1u);
        if
          (((occupied[c >> 5u] >> (c & 31u)) & 1u) != 0u || !winning(
            lines[a] | (1u << (cpos + shift)),
            shift,
            cpos,
            inp.rule,
          ))
        {
          continue;
        }
        // Keep it in a free or stale slot unless already remembered.
        let base = (p - 1u) * 16u;
        let first = extractBits(threats, base, 8u);
        let second = extractBits(threats, base + 8u, 8u);
        if (first == c || second == c) {
          continue;
        }
        if
          (first == 255u || ((occupied[first >> 5u] >> (first & 31u)) & 1u) != 0u)
        {
          threats = insertBits(threats, c, base, 8u);
        } else if
          (second == 255u || ((occupied[second >> 5u] >> (second & 31u)) & 1u) != 0u)
        {
          threats = insertBits(threats, c, base + 8u, 8u);
        }
      }
    }
    p = 3u - p;
  }
  if (score != 0u) {
    atomicAdd(&total, score);
  }
  workgroupBarrier();
  if (lane == 0u) {
    scores[group.y * grid.x + group.x] = atomicLoad(&total);
  }
}
