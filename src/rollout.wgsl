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

// The game being simulated: per line, black in bits 0-14 and white in 16-30.
var<private> rows: array<u32, 15>;
var<private> cols: array<u32, 15>;
var<private> descending: array<u32, 29>;
var<private> ascending: array<u32, 29>;
var<private> occupied: array<u32, 8>;

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

// The `winning` rule for player `q`: 1 when only exactly five wins, which in
// renju (rule 2) applies to black alone.
fn rule_for(rule: u32, q: u32) -> u32 {
  return select(0u, 1u, rule == 1u || (rule == 2u && q == 1u));
}

const DXS = array<i32, 4>(1, 0, 1, 1);
const DYS = array<i32, 4>(0, 1, 1, -1);

// The line through (x, y) along axis `a`, and the position along it.
fn line_at(x: u32, y: u32, a: u32) -> u32 {
  switch a {
    case 0u: {
      return rows[y];
    }
    case 1u: {
      return cols[x];
    }
    case 2u: {
      return descending[14u + x - y];
    }
    default: {
      return ascending[x + y];
    }
  }
}

fn pos_at(x: u32, y: u32, a: u32) -> u32 {
  return select(x, y, a == 1u);
}

// Consecutive stones through `pos` in a 15-bit mask holding it.
fn run_length(mask: u32, pos: u32) -> u32 {
  return
    countTrailingZeros(~(mask >> pos))
      + countLeadingZeros(~(mask << (31u - pos)))
      - 1u;
}

// Flip a black stone at point `c` on every line and the occupancy bits.
fn toggle_black(c: u32) {
  let x = c % 15u;
  let y = c / 15u;
  rows[y] ^= 1u << x;
  cols[x] ^= 1u << y;
  descending[14u + x - y] ^= 1u << x;
  ascending[x + y] ^= 1u << x;
  occupied[c >> 5u] ^= 1u << (c & 31u);
}

// Per axis, up to four points (bytes, 255 for none) that would turn the three
// found by `shape` into a straight four.
var<private> three_gaps: array<u32, 4>;

// The shape the black stone already at `c` makes, as rules::forbidden_in
// before its recursion: 0 for legal, 1 for forbidden, or 2 when at least two
// axes hold threes, whose straight-four points are left in `three_gaps`.
fn shape(c: u32) -> u32 {
  let x = c % 15u;
  let y = c / 15u;
  var black: array<u32, 4>;
  var pos: array<u32, 4>;
  var wide = false;
  var many = 0u;
  for (var a = 0u; a < 4u; a++) {
    pos[a] = pos_at(x, y, a);
    black[a] = line_at(x, y, a) & 32767u;
    let n = countOneBits(((black[a] << 5u) >> pos[a]) & 2047u) - 1u;
    wide = wide || n >= 5u;
    many += select(0u, 1u, n >= 2u);
  }
  // An overline needs five more black stones on one axis, and every double
  // needs at least two more on each of two axes.
  if (!wide && many < 2u) {
    return 0u;
  }
  // A five wins even when the same move also makes a forbidden shape.
  var long = false;
  for (var a = 0u; a < 4u; a++) {
    let r = run_length(black[a], pos[a]);
    if (r == 5u) {
      return 0u;
    }
    long = long || r > 5u;
  }
  if (long) {
    return 1u;
  }
  var fours = 0u;
  var candidates = 0u;
  for (var a = 0u; a < 4u; a++) {
    // Bit k of each window is the cell k - 5 steps along the axis.
    let stones = ((black[a] << 5u) >> pos[a]) & 2047u;
    var open = 0u;
    for (var k = 0u; k < 11u; k++) {
      let cx = i32(x) + DXS[a] * (i32(k) - 5);
      let cy = i32(y) + DYS[a] * (i32(k) - 5);
      if (cx < 0 || cx > 14 || cy < 0 || cy > 14) {
        continue;
      }
      let i = u32(cy) * 15u + u32(cx);
      if (((occupied[i >> 5u] >> (i & 31u)) & 1u) == 0u) {
        open |= 1u << k;
      }
    }
    // Fours: five-cell windows through the stone with four black stones and
    // an empty gap that makes exactly five on this axis.
    var formations: array<u32, 5>;
    var count = 0u;
    for (var s = 1u; s <= 5u; s++) {
      let own = (stones >> s) & 31u;
      let gaps = (open >> s) & 31u;
      if ((own | gaps) != 31u || countOneBits(own) != 4u) {
        continue;
      }
      let formation = own << s;
      var seen = false;
      for (var f = 0u; f < count; f++) {
        seen = seen || formations[f] == formation;
      }
      let d = i32(countTrailingZeros(gaps) + s) - 5;
      let gpos = u32(i32(pos[a]) + d);
      if (!seen && run_length(black[a] | (1u << gpos), gpos) == 5u) {
        formations[count] = formation;
        count++;
      }
    }
    fours += count;
    // An open three fills its gap to a straight four: four consecutive stones
    // whose two open ends each complete exactly five, without an overline.
    var found = 0u;
    three_gaps[a] = 0xffffffffu;
    for (var s = 2u; s <= 5u; s++) {
      let gaps = (open >> s) & 15u;
      if
        (countOneBits((stones >> s) & 15u) == 3u
          && countOneBits(gaps) == 1u
          && ((open >> (s - 1u)) & 1u) != 0u
          && ((open >> (s + 4u)) & 1u) != 0u
          && ((stones >> (s - 2u)) & 1u) == 0u
          && ((stones >> (s + 5u)) & 1u) == 0u)
      {
        let d = i32(countTrailingZeros(gaps) + s) - 5;
        let g = u32(i32(y) + DYS[a] * d) * 15u + u32(i32(x) + DXS[a] * d);
        three_gaps[a] = insertBits(three_gaps[a], g, found * 8u, 8u);
        found++;
      }
    }
    candidates += select(0u, 1u, found > 0u);
  }
  if (fours >= 2u) {
    return 1u;
  }
  return select(0u, 2u, candidates >= 2u);
}

// Frames of the forbidden-move search, one per stone tried: the point in bits
// 0-7, the axis in 8-10, the gap on it in 11-13 and real threes in 14-15. A
// frame places a new stone, so the depth stays below the empty points.
var<private> frames: array<u32, 225>;

// Whether a black stone at the empty (x, y) is forbidden, as rules::forbidden.
// WGSL has no recursion, so the formal check that each three's straight four
// can legally be made runs depth first on an explicit stack.
fn forbidden(x: u32, y: u32) -> bool {
  let root = y * 15u + x;
  toggle_black(root);
  frames[0] = root;
  var depth = 0u;
  var verdict = shape(root);
  loop {
    if (verdict < 2u) {
      if (depth == 0u) {
        break;
      }
      // Back up a finished check of a straight-four point to its three.
      toggle_black(frames[depth] & 255u);
      depth--;
      var frame = frames[depth];
      shape(frame & 255u);
      if (verdict == 0u) {
        // A real three: move to the next axis.
        let real = extractBits(frame, 14u, 2u) + 1u;
        if (real >= 2u) {
          verdict = 1u;
          continue;
        }
        frame = insertBits(frame, real, 14u, 2u);
        frame = insertBits(frame, extractBits(frame, 8u, 3u) + 1u, 8u, 3u);
        frame = insertBits(frame, 0u, 11u, 3u);
      } else {
        frame = insertBits(frame, extractBits(frame, 11u, 3u) + 1u, 11u, 3u);
      }
      frames[depth] = frame;
      verdict = 2u;
    }
    // Find the next straight-four point to try, unless too few axes remain.
    let frame = frames[depth];
    var a = extractBits(frame, 8u, 3u);
    var f = extractBits(frame, 11u, 3u);
    let real = extractBits(frame, 14u, 2u);
    var g = 255u;
    while (a < 4u) {
      if (f < 4u) {
        g = extractBits(three_gaps[a], f * 8u, 8u);
      }
      if (g != 255u) {
        break;
      }
      a++;
      f = 0u;
    }
    var left = 0u;
    for (var b = a; b < 4u; b++) {
      left += select(0u, 1u, (three_gaps[b] & 255u) != 255u);
    }
    if (real + left < 2u) {
      verdict = 0u;
      continue;
    }
    frames[depth] = insertBits(insertBits(frame, a, 8u, 3u), f, 11u, 3u);
    depth++;
    frames[depth] = g;
    toggle_black(g);
    verdict = shape(g);
  }
  toggle_black(root);
  return verdict == 1u;
}

// Whether this pipeline may play renju; without it the forbidden checks compile out.
override renju_rule: bool = false;
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
  rows = inp.rows;
  cols = inp.cols;
  descending = inp.descending;
  ascending = inp.ascending;
  var empty = inp.empty;
  var n = inp.count;
  var s = inp.seed ^ (inp.offset + id);
  if (s == 0u) {
    s = 1u;
  }
  occupied = inp.occupied;
  let renju = renju_rule && inp.rule == 2u;
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
      // longer line spoil it. Black never plays a forbidden point.
      let cx = c % 15u;
      let cy = c / 15u;
      let rule = rule_for(inp.rule, q);
      if
        ((rule != 1u || wins_at(
          rows[cy],
          cols[cx],
          descending[14u + cx - cy],
          ascending[cx + cy],
          cx,
          cy,
          q,
          rule,
        )) && !(renju && p == 1u && forbidden(cx, cy)))
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
      if (renju && p == 1u) {
        // Try the remaining empty points before declaring that black has no
        // legal move, which loses.
        var delta = 0u;
        for (; delta < n; delta++) {
          let k = (j + delta) % n;
          let c = extractBits(empty[k >> 2u], (k & 3u) * 8u, 8u);
          if (!forbidden(c % 15u, c / 15u)) {
            j = k;
            break;
          }
        }
        if (delta == n) {
          score = select(0u, 65538u, inp.root == 2u);
          break;
        }
      }
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
    let rule = rule_for(inp.rule, p);
    if
      (winning(rows[y], shift, x, rule)
        || winning(cols[x], shift, y, rule)
        || winning(descending[down], shift, x, rule)
        || winning(ascending[up], shift, x, rule))
    {
      score = select(0u, 65538u, p == inp.root);
      break;
    }
    // New winning points lie within four cells on a line crowded by the move,
    // and only that line can gain a five through them.
    let lines =
      array<u32, 4>(rows[y], cols[x], descending[down], ascending[up]);
    let positions = array<u32, 4>(x, y, x, x);
    for (var a = 0u; a < 4u && inp.policy != 0u; a++) {
      if (!crowded(lines[a], shift, positions[a])) {
        continue;
      }
      for (var d = -4; d <= 4; d++) {
        let cx = i32(x) + DXS[a] * d;
        let cy = i32(y) + DYS[a] * d;
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
            rule,
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
