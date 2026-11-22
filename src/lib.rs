use wasm_bindgen::prelude::*;
mod gpu;
mod lines;
mod vcf;
mod vct;
mod windows;
use windows::Windows;
type Board = [u8; 225];

// xorshift32 spans the nonzero u32 values. Reject the incomplete prefix.
const REJECTION_THRESHOLD: [u8; 226] = {
    let mut table = [0; 226];
    let mut n = 1;
    while n < table.len() {
        table[n] = (u32::MAX % n as u32) as u8;
        n += 1;
    }
    table
};

mod rules;
const AXES: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
fn run(b: &Board, i: usize, p: u8, (dx, dy): (i32, i32)) -> u8 {
    let mut count = 1;
    for sign in [-1, 1] {
        let (mut x, mut y) = ((i % 15) as i32, (i / 15) as i32);
        loop {
            x += dx * sign;
            y += dy * sign;
            if !(0..15).contains(&x) || !(0..15).contains(&y) || b[(y * 15 + x) as usize] != p {
                break;
            }
            count += 1;
        }
    }
    count
}
fn won(b: &Board, i: usize, p: u8, rule: u8) -> bool {
    AXES.iter().any(|&axis| {
        let count = run(b, i, p, axis);
        if rule == 1 || (rule == 2 && p == 1) {
            count == 5
        } else {
            count >= 5
        }
    })
}
#[wasm_bindgen]
pub fn legal_move(input: &[u8], i: usize, p: u8, rule: u8) -> bool {
    let Ok(board) = <&Board>::try_from(input) else {
        return false;
    };
    i < 225 && matches!(p, 1 | 2) && rule <= 2 && board[i] == 0 && rules::legal(board, i, p, rule)
}
#[wasm_bindgen]
pub fn black_can_move(input: &[u8]) -> bool {
    let Ok(board) = <&Board>::try_from(input) else {
        return false;
    };
    (0..225).any(|i| board[i] == 0 && rules::legal(board, i, 1, 2))
}
#[cfg(test)]
#[derive(Default)]
struct Threat {
    winning_replies: Vec<usize>,
    value: u32,
}

// Reference scan for the window counts: score every unblocked five-cell window
// completed or advanced by this move.
#[cfg(test)]
fn threat(board: &Board, point: usize, player: u8, rule: u8) -> Threat {
    let mut result = Threat::default();
    for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
        for start in -4..=0 {
            let mut stones = 0;
            let mut gap = 0;
            for step in start..start + 5 {
                let x = (point % 15) as i32 + dx * step;
                let y = (point / 15) as i32 + dy * step;
                if !(0..15).contains(&x) || !(0..15).contains(&y) {
                    stones = 0;
                    break;
                }
                let cell = (y * 15 + x) as usize;
                if cell == point || board[cell] == player {
                    stones += 1;
                } else if board[cell] == 0 {
                    gap = cell;
                } else {
                    stones = 0;
                    break;
                }
            }
            result.value += [0, 0, 1, 12, 120, 10000][stones];
            if stones == 4 && !result.winning_replies.contains(&gap) {
                let legal_win = if rule == 0 || (rule == 2 && player == 2) {
                    true // The five-cell window is already a proof in overline-allowed play.
                } else {
                    let mut after = *board;
                    after[point] = player;
                    won(&after, gap, player, rule) && rules::legal(&after, gap, player, rule)
                };
                if legal_win {
                    result.winning_replies.push(gap);
                }
            }
        }
    }
    result
}
// Return a winner only for short forcing sequences proven by legal winning points.
#[wasm_bindgen]
pub fn forced_winner(input: &[u8], player: u8, rule: u8) -> u8 {
    let Ok(board) = <&Board>::try_from(input) else {
        return 0;
    };
    if !matches!(player, 1 | 2) || rule > 2 {
        return 0;
    }
    let opponent = 3 - player;
    let windows = Windows::new(board);
    let mut enemy_wins = 0;
    for i in (0..225).filter(|&i| board[i] == 0) {
        if windows.wins(board, i, player, rule) && rules::legal(board, i, player, rule) {
            return player;
        }
        if windows.wins(board, i, opponent, rule) && rules::legal(board, i, opponent, rule) {
            enemy_wins += 1;
        }
    }
    if enemy_wins >= 2 {
        return opponent;
    }
    if enemy_wins == 0
        && (0..225).any(|i| {
            board[i] == 0
                && windows.threat(board, i, player, rule).fork
                && rules::legal(board, i, player, rule)
        })
    {
        return player;
    }
    0
}

// Root children expanded in candidate order before UCT takes over.
const ROOT_BREADTH: usize = 32;
// VCF search budgets in attacker nodes: once at the root, and for the player to
// move at each node the search expands.
const ROOT_VCF_NODES: u32 = 20000;
const NODE_VCF_NODES: u32 = 200;
// VCT (df-pn) nodes at the root: about 10 ms on average in a played game.
const ROOT_VCT_NODES: u32 = 2000;
// Strength comparisons can switch back to the earlier heuristic defense filter,
// or to 32 truncated candidates that are all expanded before UCT selection.
#[cfg(test)]
thread_local! {
    static LEGACY_DEFENSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LEGACY_BREADTH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LEGACY_TRANSPOSITIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LEGACY_VCF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LEGACY_VCT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
fn legacy_vct() -> bool {
    LEGACY_VCT.with(|flag| flag.get())
}
#[cfg(not(test))]
fn legacy_vct() -> bool {
    false
}
// Strength comparisons can turn VCF searches off.
#[cfg(test)]
fn legacy_vcf() -> bool {
    LEGACY_VCF.with(|flag| flag.get())
}
#[cfg(not(test))]
fn legacy_vcf() -> bool {
    false
}
#[cfg(test)]
fn legacy_transpositions() -> bool {
    LEGACY_TRANSPOSITIONS.with(|flag| flag.get())
}
#[cfg(not(test))]
fn legacy_transpositions() -> bool {
    false
}
#[cfg(test)]
fn legacy_defense() -> bool {
    LEGACY_DEFENSE.with(|flag| flag.get())
}
#[cfg(test)]
fn legacy_breadth() -> bool {
    LEGACY_BREADTH.with(|flag| flag.get())
}
#[cfg(not(test))]
fn legacy_defense() -> bool {
    false
}
#[cfg(not(test))]
fn legacy_breadth() -> bool {
    false
}
#[cfg(test)]
fn moves(b: &Board, p: u8, rule: u8) -> Vec<usize> {
    candidates(b, p, rule).0
}
// The flag marks a complete forced set: immediate wins, or every block of an
// immediate enemy win. Any omitted move loses at once, so the set can prove a loss.
// Candidates with their heuristic scores, for search priors; forced sets of
// wins, blocks or a fork carry no scores (all zero).
#[cfg(test)]
fn scored_candidates(b: &Board, p: u8, rule: u8) -> (Vec<usize>, bool, Vec<u32>) {
    scored_candidates_with(b, &mut Windows::new(b), p, rule)
}
// The same with the window counts of `b`, left unchanged on return.
fn scored_candidates_with(
    b: &Board,
    windows: &mut Windows,
    p: u8,
    rule: u8,
) -> (Vec<usize>, bool, Vec<u32>) {
    // Build the union of occupied neighborhoods once. Every immediate five and
    // every fork must include a stone within two squares of the candidate.
    let mut neighborhoods = [0u16; 15];
    for (i, &cell) in b.iter().enumerate() {
        if !matches!(cell, 1 | 2) {
            continue;
        }
        let (x, y) = (i % 15, i / 15);
        let columns = ((31u32 << (x + 2)) >> 4) as u16;
        for row in &mut neighborhoods[y.saturating_sub(2)..=(y + 2).min(14)] {
            *row |= columns;
        }
    }
    let mut wins = vec![];
    let mut blocks = vec![];
    let mut near = vec![];
    for (y, mut columns) in neighborhoods.into_iter().enumerate() {
        columns &= 0x7fff;
        while columns != 0 {
            let x = columns.trailing_zeros() as usize;
            columns &= columns - 1;
            let i = y * 15 + x;
            if b[i] != 0 || !rules::legal(b, i, p, rule) {
                continue;
            }
            let scan = windows.scan(i, p);
            // An open window of four is a five; exact-five rules check the line.
            let exact = |q: u8| rule == 0 || (rule == 2 && q == 2) || won(b, i, q, rule);
            if scan.five[0] && exact(p) {
                wins.push(i);
            }
            if scan.five[1] && exact(3 - p) && rules::legal(b, i, 3 - p, rule) {
                blocks.push(i);
            }
            near.push((i, scan));
        }
    }
    if !wins.is_empty() {
        let zeros = vec![0; wins.len()];
        return (wins, true, zeros);
    }
    if !blocks.is_empty() {
        let zeros = vec![0; blocks.len()];
        return (blocks, true, zeros);
    }
    let mut ranked = Vec::new();
    let mut enemy_forks = Vec::new();
    for &(i, scan) in &near {
        // Winning replies need an open four, and a fork needs two of them.
        let mut four = None;
        if scan.fours[0] > 0 {
            let attack = windows.threat(b, i, p, rule);
            if attack.fork {
                return (vec![i], false, vec![0]);
            }
            four = attack.four;
        }
        if scan.fours[1] > 1
            && windows.threat(b, i, 3 - p, rule).fork
            && rules::legal(b, i, 3 - p, rule)
        {
            enemy_forks.push(i);
        }
        ranked.push((i, four, scan.value));
    }
    // Facing an enemy fork move (a point giving two winning replies), the only
    // defenses are counter-fours and moves that leave no fork. Under the
    // five-or-more rule a defending stone never creates enemy fives, so any other
    // move loses to the fork at once and the defenses form a complete set.
    let mut complete = false;
    let mut all_lose = false;
    let legacy = legacy_defense();
    if !enemy_forks.is_empty() {
        let mut after = *b;
        let defenses: Vec<_> = ranked
            .iter()
            .copied()
            .filter(|&(i, four, _)| {
                if four.is_some() && !legacy {
                    return true;
                }
                after[i] = p;
                windows.place(i, p);
                let stops = enemy_forks.iter().all(|&j| {
                    j == i
                        || (legacy && four.is_some_and(|reply| reply != j))
                        || !rules::legal(&after, j, 3 - p, rule)
                        || !windows.threat(&after, j, 3 - p, rule).fork
                });
                windows.remove(i, p);
                after[i] = 0;
                stops
            })
            .collect();
        // Without a defense every move loses, so the whole set is lost.
        if defenses.is_empty() {
            all_lose = rule == 0 && !legacy;
        } else {
            complete = rule == 0 && !legacy;
            ranked = defenses;
        }
    }
    ranked.sort_by_key(|&(i, _, value)| {
        (
            std::cmp::Reverse(value),
            ((i % 15) as i32 - 7).pow(2) + ((i / 15) as i32 - 7).pow(2),
            i,
        )
    });
    let mut values: Vec<u32> = ranked.iter().map(|&(_, _, value)| value).collect();
    let mut near: Vec<usize> = ranked.into_iter().map(|(i, _, _)| i).collect();
    if near.is_empty() {
        near.extend((0..225).filter(|&i| b[i] == 0 && rules::legal(b, i, p, rule)));
        values = vec![0; near.len()];
    }
    // Candidates are never truncated. Only a defense set is complete, since the
    // general set omits points more than two squares from every stone.
    if legacy_breadth() && !(complete && near.len() <= 64) {
        near.truncate(32);
        values.truncate(32);
        return (near, all_lose, values);
    }
    (near, complete || all_lose, values)
}
#[cfg(test)]
fn candidates(b: &Board, p: u8, rule: u8) -> (Vec<usize>, bool) {
    let (points, forced, _) = scored_candidates(b, p, rule);
    (points, forced)
}
// Random keys for each point and colour. A position's key XORs the keys of its
// stones, so the order of moves that reached it does not matter.
const ZOBRIST: [[u64; 2]; 225] = {
    let mut table = [[0; 2]; 225];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut k = 0;
    while k < 450 {
        // splitmix64
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        table[k / 2][k % 2] = z ^ (z >> 31);
        k += 1;
    }
    table
};
fn zobrist(board: &Board) -> u64 {
    (0..225)
        .filter(|&i| matches!(board[i], 1 | 2))
        .fold(0, |key, i| key ^ ZOBRIST[i][usize::from(board[i] - 1)])
}
fn points_of(choices: Vec<usize>) -> Box<[u8]> {
    choices.into_iter().map(|i| i as u8).collect()
}
// One root move's statistics, from one tree or merged from several.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RootMove {
    i: usize,
    n: u32,
    points: u32,
    proven: Option<u8>,
    dist: u16,
    // Rollouts through the move: games, root wins, draws.
    outcomes: [u32; 3],
}
const STAT_WORDS: usize = 8;
// Read `Search::stats` words: point, n, points, proof code, plies, outcomes.
fn root_moves_of(stats: &[u32]) -> Vec<RootMove> {
    stats
        .as_chunks::<STAT_WORDS>()
        .0
        .iter()
        .map(|w| RootMove {
            i: w[0] as usize,
            n: w[1],
            points: w[2],
            proven: w[3].checked_sub(1).map(|v| v as u8),
            dist: w[4] as u16,
            outcomes: [w[5], w[6], w[7]],
        })
        .collect()
}
fn stats_of(moves: &[RootMove]) -> Vec<u32> {
    moves
        .iter()
        .flat_map(|m| {
            [
                m.i as u32,
                m.n,
                m.points,
                m.proven.map_or(0, |v| u32::from(v) + 1),
                u32::from(m.dist),
                m.outcomes[0],
                m.outcomes[1],
                m.outcomes[2],
            ]
        })
        .collect()
}
// Unproven moves with at least this share of the most simulations compete on
// their score; the rest follow by simulations.
const SCORE_SHARE: f64 = 0.15;
// Order moves best first for the side choosing them: the root player when
// `mine`, else the opponent, whose proofs and scores are reversed. Proven wins
// lead, quickest first, and proven losses trail, longest first. Between them,
// moves with enough simulations rank by a lower confidence bound on their
// score, so a move that overtakes late leads before its count catches up.
// Each leaf scores `batch` correlated rollouts, so the bound counts leaves,
// not games. Unsearched moves come last.
fn ranked(moves: &[RootMove], batch: u32, mine: bool) -> Vec<RootMove> {
    let proof = |m: &RootMove| m.proven.map(|v| if mine { v } else { 2 - v });
    let open = |m: &RootMove| m.n > 0 && !matches!(proof(m), Some(0 | 2));
    let most = moves
        .iter()
        .filter(|m| open(m))
        .map(|m| m.n)
        .max()
        .unwrap_or(0);
    let tier = |m: &RootMove| match proof(m) {
        _ if m.n == 0 => 0,
        Some(2) => 4,
        Some(0) => 1,
        _ if f64::from(m.n) >= SCORE_SHARE * f64::from(most) => 3,
        _ => 2,
    };
    let score = |m: &RootMove| {
        let q = f64::from(m.points) / (2. * f64::from(m.n));
        if mine {
            q
        } else {
            1. - q
        }
    };
    // The 5% quantile of a Beta(1, 1) posterior, by normal approximation.
    let lower = |m: &RootMove| {
        let leaves = f64::from(m.n) / f64::from(batch.max(1));
        let (a, b) = (1. + score(m) * leaves, 1. + (1. - score(m)) * leaves);
        a / (a + b) - 1.645 * (a * b / ((a + b) * (a + b) * (a + b + 1.))).sqrt()
    };
    let mut moves = moves.to_vec();
    moves.sort_by(|a, b| {
        tier(b)
            .cmp(&tier(a))
            .then_with(|| match tier(a) {
                4 => a.dist.cmp(&b.dist),
                1 => b.dist.cmp(&a.dist),
                3 => lower(b).total_cmp(&lower(a)),
                _ => std::cmp::Ordering::Equal,
            })
            .then(b.n.cmp(&a.n))
            .then_with(|| score(b).total_cmp(&score(a)))
    });
    moves
}
fn best_of(moves: &[RootMove], batch: u32) -> i32 {
    ranked(moves, batch, true)
        .first()
        .filter(|m| m.n > 0)
        .map_or(-1, |m| m.i as i32)
}
// Order `Search::stats` words best first for the root player (`mine`) or for
// its opponent, as the AI would choose: for hints and the candidate table.
#[wasm_bindgen]
pub fn rank_stats(stats: &[u32], batch: u32, mine: bool) -> Vec<u32> {
    stats_of(&ranked(&root_moves_of(stats), batch, mine))
}
fn sample_of(moves: &[RootMove], temperature: f64, unit: f64, batch: u32) -> i32 {
    let moves: Vec<&RootMove> = moves.iter().filter(|m| m.n > 0).collect();
    let lost = |m: &&RootMove| m.proven == Some(0);
    // Proofs outrank temperature: take the quickest proven win at any
    // difficulty, and the longest loss when every move loses.
    if temperature.is_nan()
        || temperature <= 0.
        || moves.iter().any(|m| m.proven == Some(2))
        || moves.iter().all(lost)
    {
        return best_of(&moves.into_iter().copied().collect::<Vec<_>>(), batch);
    }
    let moves: Vec<&RootMove> = moves.into_iter().filter(|m| !lost(m)).collect();
    let most = moves.iter().map(|m| m.n).max().unwrap_or(1) as f64;
    let weights: Vec<f64> = moves
        .iter()
        // Relative to the most visited move, so large counts cannot overflow.
        .map(|m| ((m.n as f64 / most).ln() / temperature).exp())
        .collect();
    let mut target = unit.clamp(0., 1.) * weights.iter().sum::<f64>();
    for (m, weight) in moves.iter().zip(&weights) {
        if target < *weight {
            return m.i as i32;
        }
        target -= weight;
    }
    moves.last().map_or(-1, |m| m.i as i32)
}
// Combine the root statistics of independent trees on the same position
// (root parallelization): counts add up, and any tree's proof holds, taking
// the quickest win and the longest loss or draw.
#[wasm_bindgen]
pub fn merge_stats(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut moves = root_moves_of(a);
    for m in root_moves_of(b) {
        let Some(old) = moves.iter_mut().find(|old| old.i == m.i) else {
            moves.push(m);
            continue;
        };
        old.n += m.n;
        old.points += m.points;
        for k in 0..3 {
            old.outcomes[k] += m.outcomes[k];
        }
        let rank = |p: Option<u8>| match p {
            Some(2) => 3,
            Some(0) => 2,
            Some(_) => 1,
            None => 0,
        };
        if rank(m.proven) > rank(old.proven) {
            old.proven = m.proven;
            old.dist = m.dist;
        } else if m.proven.is_some() && m.proven == old.proven {
            old.dist = if m.proven == Some(2) {
                old.dist.min(m.dist)
            } else {
                old.dist.max(m.dist)
            };
        }
    }
    stats_of(&moves)
}
// Choose a move from merged statistics, as `Search::sample_move` would.
#[wasm_bindgen]
pub fn pick_move(stats: &[u32], temperature: f64, unit: f64, batch: u32) -> i32 {
    sample_of(&root_moves_of(stats), temperature, unit, batch)
}
// A node is a position, shared by every move order that reaches it, so the
// tree is a directed acyclic graph. `children[k]` is the node after `choices[k]`.
#[derive(Default)]
struct Node {
    n: u32,
    points: u32,
    simulations: u32,
    wins: u32,
    draws: u32,
    // Candidate points, one byte each: expanded nodes far outnumber the rest,
    // and a middle-game node holds about 180 candidates.
    choices: Option<Box<[u8]>>,
    // Search priors of the candidates in 1/65535 units, used by PUCT.
    priors: Box<[u16]>,
    forced: bool,
    // Game value proven by the solver, as a doubled root score (0 loss, 1 draw, 2 win).
    proven: Option<u8>,
    // Plies from this node to the end of a proven game: the winner hurries, the loser delays.
    dist: u16,
    children: Vec<usize>,
}
#[wasm_bindgen]
pub struct Search {
    board: Board,
    root: u8,
    rule: u8,
    seed: u32,
    explore: f64,
    // Off only in strength comparisons against plain MCTS.
    solver: bool,
    nodes: Vec<Node>,
    path: Vec<usize>,
    leaf: Board,
    turn: u8,
    rounds: u32,
    depth: usize,
    // Simulations in the latest batch, the virtual count of an unexpanded child.
    unit: u32,
    // Leaves selected for one parallel batch, each holding a virtual loss.
    pending: Vec<Pending>,
    // Batches handed to the GPU and not yet scored, oldest first. Their
    // virtual losses stay on the tree while the next batch is selected.
    queued: std::collections::VecDeque<Vec<Pending>>,
    // Transposition table: position key to node.
    table: std::collections::HashMap<u64, usize>,
    root_key: u64,
    // The key of the selected leaf.
    key: u64,
    // The latest expansion (parent, key, whether it created the node), for cancel_selection.
    expansion: Option<(usize, u64, bool)>,
    // A root move already played: selection stays below it until `reroot`.
    focus: Option<usize>,
    // Rollouts finish and block fives instead of playing uniformly at random.
    policy: bool,
    // PUCT exploration constant; 0 selects plain UCT with `explore`.
    puct: f64,
    // Priors follow (score + 1) ^ prior_power.
    prior_power: f64,
    // The root player's quickest VCF, else a VCT under the five-or-more rule:
    // first move and plies to the five.
    vcf_move: Option<(usize, u16)>,
    // Whether `vcf_move` came from the VCT search.
    by_vct: bool,
}
struct Pending {
    path: Vec<usize>,
    leaf: Board,
    turn: u8,
    unit: u32,
    // The round whose seeds score this leaf, fixed when it is selected.
    round: u32,
}
#[wasm_bindgen]
impl Search {
    #[wasm_bindgen(constructor)]
    pub fn new(
        input: &[u8],
        root: u8,
        seed: u32,
        explore: f64,
        rule: Option<u8>,
    ) -> Result<Search, JsError> {
        let rule = rule.unwrap_or(0);
        let board: Board = input.try_into().map_err(|_| JsError::new("棋盤大小錯誤"))?;
        let black = board.iter().filter(|&&v| v == 1).count();
        let white = board.iter().filter(|&&v| v == 2).count();
        if !explore.is_finite()
            || explore < 0.
            || rule > 2
            || board.iter().any(|&v| v > 3)
            || !((root == 1 && black == white) || (root == 2 && black == white + 1))
            || !board.contains(&0)
            || board
                .iter()
                .enumerate()
                .any(|(i, &p)| matches!(p, 1 | 2) && won(&board, i, p, rule))
        {
            return Err(JsError::new("無效局面或參數"));
        }
        Ok((Self {
            board,
            root,
            rule,
            seed,
            explore,
            solver: true,
            nodes: vec![Node::default()],
            path: vec![],
            leaf: board,
            turn: root,
            rounds: 0,
            depth: 0,
            unit: 64,
            pending: Vec::new(),
            queued: Default::default(),
            table: std::collections::HashMap::from([(zobrist(&board), 0)]),
            root_key: zobrist(&board),
            key: zobrist(&board),
            expansion: None,
            focus: None,
            policy: false,
            puct: 0.,
            prior_power: 1.5,
            vcf_move: None,
            by_vct: false,
        })
        .with_root_choices())
    }
    pub fn direct(&self) -> i32 {
        if self.board[112] == 0 && !self.board.iter().any(|&v| matches!(v, 1 | 2)) {
            return 112;
        }
        let m = self.nodes[0].choices.as_ref().unwrap();
        if !m.is_empty() && (m.len() == 1 || won(&self.board, m[0].into(), self.root, self.rule)) {
            m[0].into()
        } else {
            -1
        }
    }
    // Return doubled terminal score or -1 when this leaf needs rollouts.
    pub fn select(&mut self) -> i32 {
        self.path.clear();
        self.path.push(0);
        self.leaf = self.board;
        self.turn = self.root;
        self.key = self.root_key;
        self.expansion = None;
        loop {
            let idx = *self.path.last().unwrap();
            let parent = &mut self.nodes[idx];
            if parent.choices.is_none() {
                // One set of window counts serves the candidates and the VCF.
                let mut windows = Windows::new(&self.leaf);
                let (choices, forced, priors) =
                    self.expand_with(&self.leaf, &mut windows, self.turn);
                let parent = &mut self.nodes[idx];
                parent.choices = Some(choices);
                parent.forced = forced;
                parent.priors = priors;
                // A VCF for the player to move proves the node at once.
                if self.solver && !legacy_vcf() {
                    if let Some((_, plies)) =
                        vcf::vcf_with(&self.leaf, windows, self.turn, self.rule, NODE_VCF_NODES)
                    {
                        let value = if self.turn == self.root { 2 } else { 0 };
                        let node = &mut self.nodes[idx];
                        node.proven = Some(value);
                        node.dist = plies;
                        return value.into();
                    }
                }
            }
            let parent = &mut self.nodes[idx];
            let choices = parent.choices.as_ref().unwrap();
            if choices.is_empty() {
                let value = if self.rule == 2 && self.turn == 1 && self.leaf.contains(&0) {
                    if self.root == 1 {
                        0
                    } else {
                        2
                    }
                } else {
                    1
                };
                parent.proven = Some(value);
                return value.into();
            }
            let total = choices.len();
            let parent = &self.nodes[idx];
            let expanded = parent.children.len();
            // UCT over expanded children, skipping those proven lost for the
            // player to move while any other remains.
            let mover_root = self.turn == self.root;
            let lost = if mover_root { 0 } else { 2 };
            let log_parent = (parent.n as f64).ln();
            let all_lost = self.solver
                && expanded > 0
                && parent
                    .children
                    .iter()
                    .all(|&id| self.nodes[id].proven == Some(lost));
            // PUCT counts visits in batches, so the batch size does not change
            // the balance between the prior and the score.
            let unit = f64::from(self.unit);
            let sqrt_parent = (parent.n as f64 / unit).sqrt();
            let prior = |k: usize| f64::from(parent.priors.get(k).copied().unwrap_or(0)) / 65535.;
            let explore = |k: usize, n: u32| {
                if self.puct > 0. {
                    self.puct * prior(k) * sqrt_parent / (1. + n as f64 / unit)
                } else {
                    self.explore * (log_parent / f64::from(n)).sqrt()
                }
            };
            let mut best = None;
            let mut high = f64::NEG_INFINITY;
            for (k, &id) in parent.children.iter().enumerate() {
                let c = &self.nodes[id];
                if self.solver && !all_lost && c.proven == Some(lost) {
                    continue;
                }
                let q = c.points as f64 / (2. * c.n as f64);
                let score = (if mover_root { q } else { 1. - q }) + explore(k, c.n);
                if score > high {
                    high = score;
                    best = Some((k, id));
                }
            }
            // First play urgency: the next candidate counts as one batch at the
            // node's mean score (with PUCT, as unvisited with its prior), so the
            // selection rule itself trades breadth against depth. Candidates are
            // sorted by score, so the next one has the largest prior left.
            let fresh = expanded < total
                && (expanded < self.initial_breadth(idx, total) || all_lost || parent.n == 0 || {
                    let q = parent.points as f64 / (2. * parent.n as f64);
                    let urgency = (if mover_root { q } else { 1. - q })
                        + if self.puct > 0. {
                            explore(expanded, 0)
                        } else {
                            self.explore * (log_parent / unit).sqrt()
                        };
                    urgency >= high
                });
            // A focused root always continues with its played move.
            let (fresh, best) = match (idx, self.focus) {
                (0, Some(point)) => {
                    let node = &self.nodes[0];
                    let k = node
                        .choices
                        .as_ref()
                        .unwrap()
                        .iter()
                        .position(|&i| usize::from(i) == point)
                        .unwrap();
                    (k == expanded, node.children.get(k).map(|&id| (k, id)))
                }
                _ => (fresh, best),
            };
            let (k, child) = if fresh {
                let i = usize::from(self.nodes[idx].choices.as_ref().unwrap()[expanded]);
                let key = self.key ^ ZOBRIST[i][usize::from(self.turn - 1)];
                // Another move order may already have reached this position.
                let found = if legacy_transpositions() {
                    None
                } else {
                    self.table.get(&key).copied()
                };
                let id = found.unwrap_or_else(|| {
                    let id = self.nodes.len();
                    self.nodes.push(Node::default());
                    if !legacy_transpositions() {
                        self.table.insert(key, id);
                    }
                    id
                });
                self.nodes[idx].children.push(id);
                self.expansion = Some((idx, key, found.is_none()));
                (expanded, id)
            } else {
                best.unwrap()
            };
            let i = usize::from(self.nodes[idx].choices.as_ref().unwrap()[k]);
            self.key ^= ZOBRIST[i][usize::from(self.turn - 1)];
            self.path.push(child);
            self.depth = self.depth.max(self.path.len() - 1);
            if let Some(value) = self.nodes[child].proven {
                return value.into();
            }
            self.leaf[i] = self.turn;
            let player = self.turn;
            self.turn = 3 - player;
            let value = if won(&self.leaf, i, player, self.rule) {
                Some(if player == self.root { 2 } else { 0 })
            } else if !self.leaf.contains(&0) {
                Some(1)
            } else {
                None
            };
            if let Some(value) = value {
                self.nodes[child].proven = Some(value);
                return value.into();
            }
            // A transposition with statistics continues the descent.
            if fresh && self.nodes[child].n == 0 {
                return -1;
            }
        }
    }
    pub fn cpu_batch(&self, batch: u32) -> u32 {
        self.cpu_batch_slice(batch, 0, batch)
    }
    pub fn cpu_batch_slice(&self, batch: u32, start: u32, count: u32) -> u32 {
        self.cpu_batch_at(batch, start, count, self.rounds)
    }
    // Leaves selected but not yet scored, waiting or on the GPU.
    pub fn in_flight(&self) -> u32 {
        (self.pending.len() + self.queued.iter().map(Vec::len).sum::<usize>()) as u32
    }
    // Undo an unscored expansion when the player moves during a CPU batch.
    pub fn cancel_selection(&mut self) {
        if let (Some((parent, key, created)), [.., last, child]) =
            (self.expansion.take(), self.path.as_slice())
        {
            if *last == parent && self.nodes[*child].n == 0 {
                self.nodes[parent].children.pop();
                if created {
                    self.nodes.pop();
                    self.table.remove(&key);
                }
            }
        }
        self.path.clear();
    }
    // Select up to `limit` rounds for one parallel batch. Each pending leaf puts a
    // virtual loss of `batch` simulations on its path, so later selections in the
    // same batch spread to other lines. Terminal leaves are scored immediately.
    // Returns the number of pending leaves.
    pub fn select_batch(&mut self, limit: u32, batch: u32) -> u32 {
        for _ in 0..limit {
            if self.solved() >= 0 {
                break;
            }
            let terminal = self.select();
            if terminal >= 0 {
                self.update(terminal as u32, 1);
                continue;
            }
            let path = std::mem::take(&mut self.path);
            self.virtual_loss(&path, batch, true);
            // Every selection or terminal score raises rounds plus the leaves
            // in flight by one, so each leaf gets its own round.
            let round = self.rounds.wrapping_add(self.in_flight());
            self.pending.push(Pending {
                path,
                leaf: self.leaf,
                turn: self.turn,
                unit: batch,
                round,
            });
        }
        self.pending.len() as u32
    }
    // Score queued and pending leaves on the CPU in selection order, with the
    // same seeds the GPU would use.
    pub fn cpu_pending(&mut self, batch: u32) {
        self.queue_pending();
        while !self.queued.is_empty() {
            self.cpu_finish(batch);
        }
    }
    pub fn rounds(&self) -> u32 {
        self.rounds
    }
    pub fn update(&mut self, points: u32, count: u32) {
        for &id in &self.path {
            let node = &mut self.nodes[id];
            node.points += points;
            node.n += count;
        }
        self.rounds += 1;
        if self.solver {
            self.propagate_proof();
        }
    }
    // Return the proven root value (0 loss, 1 draw, 2 win) or -1. With a
    // focus, the value of the focused move instead.
    pub fn solved(&self) -> i32 {
        let node = match self.focus {
            Some(point) => match self.edges(0).find(|&(i, _)| i == point) {
                Some((_, child)) => child,
                None => return -1,
            },
            None => &self.nodes[0],
        };
        node.proven.map_or(-1, i32::from)
    }
    // Keep searching below a root move that has been played, while the
    // opponent thinks. Returns false when the move is not a candidate.
    pub fn focus(&mut self, point: usize) -> bool {
        let node = &mut self.nodes[0];
        let expanded = node.children.len();
        let Some(choices) = node.choices.as_mut() else {
            return false;
        };
        let Some(k) = choices.iter().position(|&i| usize::from(i) == point) else {
            return false;
        };
        // Unexpanded candidates may be reordered: make the move the next one.
        if k > expanded {
            choices.swap(k, expanded);
            if node.priors.len() > k {
                node.priors.swap(k, expanded);
            }
        }
        self.focus = Some(point);
        true
    }
    // Continue on the position after the focused move and the opponent's
    // reply, keeping that subtree and dropping the rest. Returns false when
    // the board is not such a position or the search never reached it.
    pub fn reroot(&mut self, input: &[u8]) -> bool {
        let Ok(board) = <Board>::try_from(input) else {
            return false;
        };
        let changed: Vec<usize> = (0..225).filter(|&i| board[i] != self.board[i]).collect();
        let [a, b] = changed[..] else {
            return false;
        };
        let (mine, reply) = if board[a] == self.root {
            (a, b)
        } else {
            (b, a)
        };
        if self.in_flight() > 0
            || self.board[mine] != 0
            || self.board[reply] != 0
            || board[mine] != self.root
            || board[reply] != 3 - self.root
            || self.focus.is_some_and(|point| point != mine)
            || won(&board, reply, 3 - self.root, self.rule)
            || !board.contains(&0)
        {
            return false;
        }
        let key = zobrist(&board);
        let Some(&start) = self.table.get(&key) else {
            return false;
        };
        // Copy the subgraph reachable from the new root, renumbering nodes.
        let mut ids = std::collections::HashMap::from([(start, 0)]);
        let mut order = vec![(start, key, self.root)];
        let mut k = 0;
        while k < order.len() {
            let (old, key, turn) = order[k];
            k += 1;
            let node = &self.nodes[old];
            for (&i, &child) in node.choices.iter().flatten().zip(&node.children) {
                let i = usize::from(i);
                if let std::collections::hash_map::Entry::Vacant(entry) = ids.entry(child) {
                    entry.insert(order.len());
                    order.push((child, key ^ ZOBRIST[i][usize::from(turn - 1)], 3 - turn));
                }
            }
        }
        let mut nodes: Vec<Node> = order
            .iter()
            .map(|&(old, _, _)| std::mem::take(&mut self.nodes[old]))
            .collect();
        for node in &mut nodes {
            for child in &mut node.children {
                *child = ids[child];
            }
        }
        if nodes[0].choices.is_none() {
            let (choices, forced, priors) = self.expand(&board, self.root);
            nodes[0].choices = Some(choices);
            nodes[0].forced = forced;
            nodes[0].priors = priors;
        }
        self.table = order
            .iter()
            .enumerate()
            .map(|(id, &(_, key, _))| (key, id))
            .collect();
        self.nodes = nodes;
        self.board = board;
        self.root_key = key;
        self.leaf = board;
        self.turn = self.root;
        self.path.clear();
        self.expansion = None;
        self.focus = None;
        self.depth = 0;
        self.root_vcf();
        true
    }
    // A batch has at most 8192 games, so scores and wins fit in separate halfwords.
    pub fn update_simulations(&mut self, tally: u32, count: u32) {
        self.unit = count.max(1);
        let wins = tally >> 16;
        let points = tally & 65535;
        let draws = points.saturating_sub(2 * wins);
        // Every node on the path keeps rollout outcomes, so a reused subtree
        // still reports them once its node becomes a root move.
        for &id in &self.path {
            let node = &mut self.nodes[id];
            node.simulations += count;
            node.wins += wins;
            node.draws += draws;
        }
        self.update(points, count);
    }
    pub fn set_policy(&mut self, policy: bool) {
        self.policy = policy;
    }
    // Use PUCT with this constant (0 for UCT). Priors of nodes expanded
    // later follow the power; an unexpanded root is scored again.
    pub fn set_puct(&mut self, puct: f64, prior_power: f64) {
        self.puct = puct.max(0.);
        self.prior_power = prior_power.max(0.);
        if self.nodes[0].children.is_empty() {
            let (_, _, priors) = self.expand(&self.board, self.root);
            self.nodes[0].priors = priors;
        }
    }
    // Simulations counted at the root, including those inherited by `reroot`.
    pub fn visits(&self) -> u32 {
        self.nodes[0].n
    }
    pub fn has_moves(&self) -> bool {
        !self.nodes[0].choices.as_ref().unwrap().is_empty()
    }
    pub fn simulation_draws(&self) -> u32 {
        self.nodes[0].draws
    }
    pub fn simulation_wins(&self) -> u32 {
        self.nodes[0].wins
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    // The first move of `ranked`: the quickest proven win, else the best
    // well-searched move, avoiding proven losses (delaying them when all lose).
    pub fn best_move(&self) -> i32 {
        match self.vcf_move.filter(|_| self.solver) {
            Some((point, _)) => point as i32,
            None => best_of(&self.root_moves(), self.unit),
        }
    }
    // Sample a root move with probability proportional to N^(1 / temperature).
    // `unit` is a uniform number in [0, 1). Temperature 0 plays `best_move`.
    // Proven losses are avoided while any other move remains.
    pub fn sample_move(&self, temperature: f64, unit: f64) -> i32 {
        // A proven win outranks temperature.
        match self.vcf_move.filter(|_| self.solver) {
            Some((point, _)) => point as i32,
            None => sample_of(&self.root_moves(), temperature, unit, self.unit),
        }
    }
    // The root player's quickest VCF as its first move and plies to the five,
    // or -1 and 0.
    pub fn vcf_move(&self) -> i32 {
        self.vcf_move.map_or(-1, |(point, _)| point as i32)
    }
    pub fn vcf_plies(&self) -> u32 {
        self.vcf_move.map_or(0, |(_, plies)| u32::from(plies))
    }
    // Whether the root's forced win needs threes (VCT) rather than fours only.
    pub fn by_vct(&self) -> bool {
        self.by_vct
    }
    pub fn move_outcomes(&self, point: usize) -> Vec<u32> {
        self.edges(0).find(|&(i, _)| i == point).map_or_else(
            || vec![0, 0, 0],
            |(_, node)| vec![node.simulations, node.wins, node.draws],
        )
    }
    pub fn stats(&self) -> Vec<u32> {
        self.stats_at(0)
    }
    // The same statistics for the opponent's replies to the focused move,
    // still from the root player's view, for hints while pondering.
    pub fn focus_stats(&self) -> Vec<u32> {
        let (Some(point), Some(id)) = (self.focus, self.focus_node()) else {
            return Vec::new();
        };
        let stats = self.stats_at(id);
        // A node proven by the opponent's VCF has no replies searched: name
        // the first move of that VCF instead.
        if stats.is_empty() && self.nodes[id].proven == Some(0) {
            let mut after = self.board;
            after[point] = self.root;
            if let Some((first, plies)) = vcf::vcf(&after, 3 - self.root, self.rule, ROOT_VCF_NODES)
                .or_else(|| vct::vct(&after, 3 - self.root, self.rule, ROOT_VCT_NODES))
            {
                // Proof code 1: a root loss, after `plies - 1` more plies.
                return vec![first as u32, 1, 0, 1, u32::from(plies - 1), 0, 0, 0];
            }
        }
        stats
    }
    // Simulations below the focused move, from the AI's search and pondering.
    pub fn focus_visits(&self) -> u32 {
        self.focus_node().map_or(0, |id| self.nodes[id].n)
    }
    fn focus_node(&self) -> Option<usize> {
        let point = self.focus?;
        let node = &self.nodes[0];
        node.choices
            .iter()
            .flatten()
            .zip(&node.children)
            .find(|(&i, _)| usize::from(i) == point)
            .map(|(_, &id)| id)
    }
    fn stats_at(&self, id: usize) -> Vec<u32> {
        self.edges(id)
            .flat_map(|(i, n)| {
                [
                    i as u32,
                    n.n,
                    n.points,
                    n.proven.map_or(0, |v| u32::from(v) + 1),
                    u32::from(n.dist),
                    n.simulations,
                    n.wins,
                    n.draws,
                ]
            })
            .collect()
    }
}
impl Search {
    fn cpu_batch_at(&self, batch: u32, start: u32, count: u32, round: u32) -> u32 {
        let lines = lines::Lines::from_board(&self.leaf);
        let (empty, count_empty) = self.rollout_empty_points();
        let threats = self.rollout_threats(&lines);
        let offset = round.wrapping_mul(batch);
        (start..start.saturating_add(count).min(batch))
            .map(|id| {
                let points = self.rollout(
                    lines.clone(),
                    empty,
                    count_empty,
                    self.seed ^ offset.wrapping_add(id),
                    threats,
                );
                points + (u32::from(points == 2) << 16)
            })
            .sum()
    }
    // Hand the selected leaves over to the GPU, keeping their virtual losses.
    fn queue_pending(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        self.queued.push_back(pending);
    }
    // Score the oldest queued batch on the CPU, as `Gpu::finish` would.
    fn cpu_finish(&mut self, batch: u32) {
        let Some(leaves) = self.queued.front() else {
            return;
        };
        let jobs: Vec<(Board, u8, u32)> =
            leaves.iter().map(|p| (p.leaf, p.turn, p.round)).collect();
        let tallies: Vec<u32> = jobs
            .into_iter()
            .map(|(leaf, turn, round)| {
                self.leaf = leaf;
                self.turn = turn;
                self.cpu_batch_at(batch, 0, batch, round)
            })
            .collect();
        self.land(&tallies, batch);
    }
    // Score the oldest queued batch, in selection order.
    fn land(&mut self, tallies: &[u32], batch: u32) {
        let leaves = self.queued.pop_front().unwrap_or_default();
        for p in &leaves {
            self.virtual_loss(&p.path, p.unit, false);
        }
        for (p, &tally) in leaves.into_iter().zip(tallies) {
            self.path = p.path;
            self.update_simulations(tally, batch);
        }
    }
    // A path looks lost for each player who chose a move on it; the root looks drawn.
    fn virtual_loss(&mut self, path: &[usize], unit: u32, add: bool) {
        for (depth, &id) in path.iter().enumerate() {
            let points = match depth {
                0 => unit,
                d if d % 2 == 1 => 0,
                _ => 2 * unit,
            };
            let node = &mut self.nodes[id];
            if add {
                node.n += unit;
                node.points += points;
            } else {
                node.n -= unit;
                node.points -= points;
            }
        }
    }
    // Score the pending leaves at once with the given tallies.
    #[cfg(test)]
    fn apply_pending(&mut self, tallies: &[u32], batch: u32) {
        assert!(self.queued.is_empty());
        self.queue_pending();
        self.land(tallies, batch);
    }
    // Children expanded in candidate order before UCT decides: one everywhere,
    // and a first slice of the strongest candidates at the root.
    fn initial_breadth(&self, idx: usize, total: usize) -> usize {
        if legacy_breadth() {
            total
        } else if idx == 0 {
            ROOT_BREADTH
        } else {
            1
        }
    }
    // Candidates of a position with their forced flag and PUCT priors.
    fn expand(&self, board: &Board, turn: u8) -> (Box<[u8]>, bool, Box<[u16]>) {
        self.expand_with(board, &mut Windows::new(board), turn)
    }
    fn expand_with(
        &self,
        board: &Board,
        windows: &mut Windows,
        turn: u8,
    ) -> (Box<[u8]>, bool, Box<[u16]>) {
        let (points, forced, values) = scored_candidates_with(board, windows, turn, self.rule);
        let weights: Vec<f64> = values
            .iter()
            .map(|&v| (f64::from(v) + 1.).powf(self.prior_power))
            .collect();
        let total: f64 = weights.iter().sum();
        let priors = weights
            .iter()
            .map(|w| ((w / total * 65535.).round() as u16).max(1))
            .collect();
        (points_of(points), forced, priors)
    }
    fn with_root_choices(mut self) -> Self {
        let (choices, forced, priors) = self.expand(&self.board, self.root);
        self.nodes[0].choices = Some(choices);
        self.nodes[0].forced = forced;
        self.nodes[0].priors = priors;
        self.root_vcf();
        self
    }
    // Look for a VCF, then a VCT, of the root player; one proves the root a win.
    fn root_vcf(&mut self) {
        self.vcf_move = None;
        self.by_vct = false;
        if !legacy_vcf() {
            self.vcf_move = vcf::vcf(&self.board, self.root, self.rule, ROOT_VCF_NODES);
            if self.vcf_move.is_none() && !legacy_vct() {
                self.vcf_move = vct::vct(&self.board, self.root, self.rule, ROOT_VCT_NODES);
                self.by_vct = self.vcf_move.is_some();
            }
        }
        if let Some((_, plies)) = self.vcf_move {
            self.nodes[0].proven = Some(2);
            self.nodes[0].dist = plies;
        }
    }
    // Expanded children of a node with the moves that reach them.
    fn edges(&self, id: usize) -> impl Iterator<Item = (usize, &Node)> {
        let node = &self.nodes[id];
        node.choices
            .iter()
            .flatten()
            .zip(&node.children)
            .map(|(&i, &child)| (usize::from(i), &self.nodes[child]))
    }
    // Simulated root moves for choosing a move; proofs only with the solver.
    fn root_moves(&self) -> Vec<RootMove> {
        self.root_children()
            .map(|(i, node)| RootMove {
                i,
                n: node.n,
                points: node.points,
                proven: node.proven.filter(|_| self.solver),
                dist: node.dist,
                outcomes: [node.simulations, node.wins, node.draws],
            })
            .collect()
    }
    fn root_children(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.edges(0).filter(|(_, child)| child.n > 0)
    }
    // Back up proofs along the last path. A player to move wins with any winning
    // child; a loss or draw needs every child proven over a complete forced set.
    fn propagate_proof(&mut self) {
        for depth in (0..self.path.len()).rev() {
            let id = self.path[depth];
            let node = &self.nodes[id];
            let (goal, pick): (u8, fn(u8, u8) -> u8) = if depth % 2 == 0 {
                (2, u8::max)
            } else {
                (0, u8::min)
            };
            let children = || node.children.iter().map(|&c| &self.nodes[c]);
            // A proven node keeps its value, but its distance follows newer proofs
            // below it: the quickest win, or the longest loss or draw.
            if let Some(value) = node.proven {
                let dists = children()
                    .filter(|c| c.proven == Some(value))
                    .map(|c| c.dist);
                let dist = if value == goal {
                    dists.min()
                } else {
                    dists.max()
                };
                if let Some(dist) = dist {
                    self.nodes[id].dist = dist.saturating_add(1);
                }
                continue;
            }
            let complete = node.forced
                && node
                    .choices
                    .as_ref()
                    .is_some_and(|c| c.len() == node.children.len());
            // Take the quickest win found, or else delay the best forced result.
            let quickest = children()
                .filter(|c| c.proven == Some(goal))
                .map(|c| c.dist)
                .min();
            let proof = if let Some(dist) = quickest {
                Some((goal, dist))
            } else if complete {
                children()
                    .try_fold(2 - goal, |acc, c| c.proven.map(|v| pick(acc, v)))
                    .map(|value| {
                        let longest = children()
                            .filter(|c| c.proven == Some(value))
                            .map(|c| c.dist)
                            .max();
                        (value, longest.unwrap_or(0))
                    })
            } else {
                None
            };
            let Some((proof, dist)) = proof else {
                break;
            };
            let node = &mut self.nodes[id];
            node.proven = Some(proof);
            node.dist = dist.saturating_add(1);
        }
    }
    fn rollout_empty_points(&self) -> ([u8; 225], usize) {
        let mut empty = [0u8; 225];
        let mut count = 0;
        for (i, &cell) in self.leaf.iter().enumerate() {
            if cell == 0 {
                empty[count] = i as u8;
                count += 1;
            }
        }
        (empty, count)
    }
    fn rollout_threats(&self, lines: &lines::Lines) -> Threats {
        rollout_threats(&self.leaf, lines, self.rule)
    }
    fn exact(&self, player: u8) -> bool {
        exact(self.rule, player)
    }
    // A rollout plays uniformly random empty points. With the policy it first
    // finishes its own five, else blocks the opponent's; each player remembers
    // up to two winning points, found at the leaf and after each of its moves.
    fn rollout(
        &self,
        mut lines: lines::Lines,
        mut empty: [u8; 225],
        mut n: usize,
        mut rng: u32,
        mut threats: Threats,
    ) -> u32 {
        let mut board = self.leaf;
        let mut p = self.turn;
        if rng == 0 {
            rng = 1
        }
        let policy = self.policy;
        while n > 0 {
            let valid = |q: u8, board: &Board| {
                threats[usize::from(q - 1)]
                    .into_iter()
                    .map(usize::from)
                    .find(|&c| {
                        // An empty winning point stays winning unless exact
                        // five rules let a longer line spoil it.
                        c != usize::from(NO_THREAT)
                            && board[c] == 0
                            && (!self.exact(q) || lines.wins_at(c, q, true))
                            && rules::legal(board, c, p, self.rule)
                    })
            };
            let forced = if policy {
                valid(p, &board).or_else(|| valid(3 - p, &board))
            } else {
                None
            };
            let j = if let Some(c) = forced {
                (0..n).find(|&k| usize::from(empty[k]) == c).unwrap()
            } else {
                loop {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    if rng > u32::from(REJECTION_THRESHOLD[n]) {
                        break;
                    }
                }
                let mut j = ((rng - 1) % n as u32) as usize;
                if self.rule == 2
                    && p == 1
                    && !rules::legal(&board, usize::from(empty[j]), p, self.rule)
                {
                    // Try the remaining empty points before declaring that black has no legal move.
                    let Some(index) = (1..n)
                        .map(|delta| (j + delta) % n)
                        .find(|&k| rules::legal(&board, usize::from(empty[k]), p, self.rule))
                    else {
                        return if self.root == 2 { 2 } else { 0 };
                    };
                    j = index;
                }
                j
            };
            let i = usize::from(empty[j]);
            n -= 1;
            empty[j] = empty[n];
            board[i] = p;
            // Black wins only with exactly five in renju.
            if lines.place_with_rule(i, p, self.exact(p)) {
                return if p == self.root { 2 } else { 0 };
            }
            // New winning points lie within four cells of the move, on a line
            // that already holds four of the mover's stones there. Only that
            // line can gain a five through them.
            if policy {
                for (a, &(dx, dy)) in AXES.iter().enumerate() {
                    if !lines.crowded(i, p, a) {
                        continue;
                    }
                    for d in (-4..=4).filter(|&d| d != 0) {
                        let x = (i % 15) as i32 + dx * d;
                        let y = (i / 15) as i32 + dy * d;
                        if !(0..15).contains(&x) || !(0..15).contains(&y) {
                            continue;
                        }
                        let c = (y * 15 + x) as usize;
                        if board[c] == 0 && lines.wins_on(c, p, self.exact(p), a) {
                            add_threat(&mut threats, &board, p, c);
                        }
                    }
                }
            }
            p = 3 - p
        }
        1
    }
}
fn exact(rule: u8, player: u8) -> bool {
    rule == 1 || (rule == 2 && player == 1)
}
// Winning points of each player at a leaf, in point order.
fn rollout_threats(board: &Board, lines: &lines::Lines, rule: u8) -> Threats {
    let mut threats = [[NO_THREAT; 2]; 2];
    for i in (0..225).filter(|&i| board[i] == 0) {
        for q in [1, 2] {
            if lines.wins_at(i, q, exact(rule, q)) {
                add_threat(&mut threats, board, q, i);
            }
        }
    }
    threats
}
// Up to two remembered winning points per player; 255 marks an empty slot.
type Threats = [[u8; 2]; 2];
const NO_THREAT: u8 = 255;
// Record a new winning point in a free or occupied (stale) slot, or drop it.
fn add_threat(threats: &mut Threats, board: &Board, player: u8, c: usize) {
    let slots = &mut threats[usize::from(player - 1)];
    if slots.contains(&(c as u8)) {
        return;
    }
    if let Some(slot) = slots
        .iter_mut()
        .find(|slot| **slot == NO_THREAT || board[usize::from(**slot)] != 0)
    {
        *slot = c as u8;
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;

    fn place(stones: &[usize], player: u8, board: &mut Board) {
        for &i in stones {
            board[i] = player;
        }
    }

    // Run CPU rounds and return the number of games counted into the root.
    fn run_rounds(s: &mut Search, rounds: u32, batch: u32) -> u32 {
        let mut total = 0;
        for _ in 0..rounds {
            let terminal = s.select();
            if terminal >= 0 {
                s.update(terminal as u32, 1);
                total += 1;
            } else {
                s.update_simulations(s.cpu_batch(batch), batch);
                total += batch;
            }
        }
        total
    }

    #[test]
    fn detects_five_in_every_direction_and_at_edges() {
        // (start x, start y, axis), covering corners and every edge.
        let lines = [
            (0, 0, (1, 0)),
            (10, 14, (1, 0)),
            (0, 0, (0, 1)),
            (14, 10, (0, 1)),
            (0, 0, (1, 1)),
            (10, 10, (1, 1)),
            (0, 14, (1, -1)),
            (10, 4, (1, -1)),
        ];
        for (x, y, (dx, dy)) in lines {
            let cells: Vec<usize> = (0..5)
                .map(|k| ((y + dy * k) * 15 + x + dx * k) as usize)
                .collect();
            let mut board = [0; 225];
            place(&cells[..4], 1, &mut board);
            for rule in 0..=2 {
                assert!(!won(&board, cells[3], 1, rule));
            }
            // The last stone may complete the line at an end or in the middle.
            for &last in &cells {
                let mut board = [0; 225];
                place(&cells, 1, &mut board);
                let mut lines = lines::Lines::from_board(&[0; 225]);
                for &i in cells.iter().filter(|&&i| i != last) {
                    lines.place(i, 1);
                }
                assert!(lines.place(last, 1));
                for rule in 0..=2 {
                    assert!(won(&board, last, 1, rule), "{x},{y} {dx},{dy} rule {rule}");
                }
            }
        }
    }

    #[test]
    fn overline_depends_on_rule() {
        let six = [105, 106, 107, 108, 109, 110];
        for player in [1, 2] {
            let mut board = [0; 225];
            place(&six, player, &mut board);
            assert!(won(&board, 110, player, 0));
            assert!(!won(&board, 110, player, 1));
            assert_eq!(won(&board, 110, player, 2), player == 2);
            let mut lines = lines::Lines::from_board(&[0; 225]);
            for &i in &six[..5] {
                lines.place(i, player);
            }
            assert!(!lines.clone().place_with_rule(110, player, true));
            assert!(lines.place_with_rule(110, player, false));
        }
        let mut board = [0; 225];
        place(&six[..5], 1, &mut board);
        assert!(!legal_move(&board, 110, 1, 2));
        assert!(legal_move(&board, 110, 1, 0));
    }

    #[test]
    fn full_board_without_five_is_a_draw() {
        // This two-colour period never has more than two equal stones in a line.
        let side = |i: usize| ((i % 15 + 2 * (i / 15)) % 4) / 2;
        let black = if (0..225).filter(|&i| side(i) == 0).count() == 113 {
            0
        } else {
            1
        };
        let last = (0..225).find(|&i| side(i) == black).unwrap();
        for rule in [0, 1] {
            let mut board = [0; 225];
            for i in (0..225).filter(|&i| i != last) {
                board[i] = if side(i) == black { 1 } else { 2 };
            }
            let mut s = Search::new(&board, 1, 7, 1., Some(rule)).unwrap();
            assert_eq!(s.direct(), last as i32);
            assert_eq!(s.select(), 1);
        }
    }

    #[test]
    fn immediate_win_beats_blocking() {
        let mut board = [0; 225];
        place(&[108, 109, 110, 111], 1, &mut board);
        place(&[123, 124, 125, 126], 2, &mut board);
        let s = Search::new(&board, 1, 1, 1., None).unwrap();
        let point = s.direct();
        assert!([107, 112].contains(&point), "{point}");
    }

    #[test]
    fn blocks_the_only_winning_point() {
        let mut board = [0; 225];
        place(&[120, 121, 122, 123], 2, &mut board);
        place(&[112, 50, 170, 200], 1, &mut board);
        let s = Search::new(&board, 1, 1, 1., None).unwrap();
        assert_eq!(s.direct(), 124);
    }

    #[test]
    fn empty_board_takes_the_center() {
        let s = Search::new(&[0; 225], 1, 1, 1., None).unwrap();
        assert_eq!(s.direct(), 112);
    }

    #[test]
    fn opponent_nodes_minimize_the_root_score() {
        let mut board = [0; 225];
        board[112] = 1;
        let mut s = Search::new(&board, 2, 1, 0., None).unwrap();
        // Moves live on the parent: `children[k]` follows `choices[k]`.
        let node = |n, points, children: Vec<usize>, choices: Option<Vec<usize>>| Node {
            n,
            points,
            children,
            choices: choices.map(points_of),
            ..Default::default()
        };
        s.nodes = vec![
            node(40, 40, vec![1, 2], Some(vec![111, 113])),
            node(20, 40, vec![3, 4], Some(vec![97, 127])),
            node(20, 0, vec![], None),
            node(10, 20, vec![], None),
            node(10, 0, vec![], None),
        ];
        // The root (white) prefers 111; black then prefers 127, the worse reply for white.
        assert_eq!(s.select(), -1);
        assert_eq!(s.path[..3], [0, 1, 4]);
    }

    #[test]
    fn root_counts_every_game_and_search_reaches_the_second_ply() {
        let mut board = [0; 225];
        board[112] = 1;
        let mut s = Search::new(&board, 2, 42, 2f64.sqrt(), None).unwrap();
        let total = run_rounds(&mut s, 80, 16);
        let root = &s.nodes[0];
        assert_eq!(root.n, total);
        let children: u32 = root.children.iter().map(|&id| s.nodes[id].n).sum();
        assert_eq!(children, total);
        assert!(s.depth() >= 2);
    }

    // Solver tests turn VCF off, which would prove these positions at once.
    fn without_vcf() {
        LEGACY_VCF.with(|flag| flag.set(true));
        LEGACY_VCT.with(|flag| flag.set(true));
    }

    #[test]
    fn root_vcf_proves_an_open_three_at_once() {
        let mut board = [0; 225];
        place(&[109, 110, 111], 1, &mut board);
        place(&[0, 14, 210], 2, &mut board);
        let s = Search::new(&board, 1, 3, 2f64.sqrt(), None).unwrap();
        assert_eq!(s.solved(), 2);
        assert_eq!(s.vcf_plies(), 3);
        assert!([108, 112].contains(&s.best_move()));
        assert_eq!(s.sample_move(2., 0.7), s.best_move());
    }

    #[test]
    fn solver_proves_an_open_four_win() {
        without_vcf();
        let mut board = [0; 225];
        place(&[109, 110, 111], 1, &mut board);
        place(&[0, 14, 210], 2, &mut board);
        let mut s = Search::new(&board, 1, 3, 2f64.sqrt(), None).unwrap();
        for _ in 0..50 {
            if s.solved() >= 0 {
                break;
            }
            run_rounds(&mut s, 1, 8);
        }
        assert_eq!(s.solved(), 2);
        let best = s.best_move();
        assert!([108, 112].contains(&best));
        // The open four, white's block, then black's five.
        let (_, child) = s.root_children().find(|c| c.0 as i32 == best).unwrap();
        assert_eq!(child.dist, 2);
    }

    // A root with hand-made children: (point, simulations, proof, plies to the end).
    fn tree(children: &[(usize, u32, Option<u8>, u16)]) -> Search {
        let mut board = [0; 225];
        board[112] = 1;
        let mut s = Search::new(&board, 2, 1, 1., None).unwrap();
        s.nodes.truncate(1);
        s.nodes[0].choices = Some(points_of(children.iter().map(|c| c.0).collect()));
        for &(_, n, proven, dist) in children {
            let id = s.nodes.len();
            s.nodes[0].children.push(id);
            s.nodes.push(Node {
                n,
                points: n,
                proven,
                dist,
                ..Default::default()
            });
        }
        s
    }

    #[test]
    fn best_move_hurries_wins_and_delays_losses() {
        let s = tree(&[
            (1, 900, Some(0), 1),
            (2, 10, Some(0), 5),
            (3, 50, Some(0), 3),
        ]);
        assert_eq!(s.best_move(), 2);
        let s = tree(&[(1, 900, None, 0), (2, 10, Some(2), 6), (3, 5, Some(2), 2)]);
        assert_eq!(s.best_move(), 3);
    }

    #[test]
    fn best_move_prefers_a_well_searched_higher_score() {
        // Scores in root wins: 52% over 4780 games, 80% over 3570, and 95%
        // over only 300, too few to compete.
        let mut s = tree(&[(1, 4780, None, 0), (2, 3570, None, 0), (3, 300, None, 0)]);
        for (id, rate) in [(1, 0.52), (2, 0.8), (3, 0.95)] {
            s.nodes[id].points = (2. * rate * f64::from(s.nodes[id].n)) as u32;
        }
        assert_eq!(s.best_move(), 2);
        assert_eq!(pick_move(&s.stats(), 0., 0., 64), 2);
        // A close score does not overturn a much larger count.
        s.nodes[2].points = (2. * 0.53 * 3570.) as u32;
        assert_eq!(s.best_move(), 1);
    }

    #[test]
    fn ranked_stats_follow_the_choosing_side() {
        let mut s = tree(&[(1, 900, None, 0), (2, 800, None, 0), (3, 50, Some(0), 2)]);
        s.nodes[1].points = 2 * 300;
        s.nodes[2].points = 2 * 600;
        let order = |mine| {
            let r = rank_stats(&s.stats(), 64, mine);
            r.chunks(STAT_WORDS).map(|w| w[0]).collect::<Vec<_>>()
        };
        // The root player loses at 3, so its opponent wins there.
        assert_eq!(order(true), [2, 1, 3]);
        assert_eq!(order(false), [3, 1, 2]);
    }

    #[test]
    fn proven_distances_follow_quicker_wins() {
        let mut s = tree(&[(1, 90, Some(2), 6), (2, 10, Some(2), 2)]);
        s.nodes[0].proven = Some(2);
        s.nodes[0].dist = 7;
        s.path = vec![0, 2];
        s.propagate_proof();
        assert_eq!(s.nodes[0].dist, 3);
    }

    #[test]
    fn merged_statistics_add_counts_and_keep_proofs() {
        let a = tree(&[(1, 90, None, 0), (2, 10, Some(2), 6), (3, 40, Some(0), 3)]);
        let b = tree(&[
            (2, 5, Some(2), 2),
            (1, 30, None, 0),
            (4, 70, None, 0),
            (3, 1, Some(0), 7),
        ]);
        let merged = root_moves_of(&merge_stats(&a.stats(), &b.stats()));
        let get = |i| *merged.iter().find(|m| m.i == i).unwrap();
        assert_eq!((get(1).n, get(1).proven), (120, None));
        // The quickest win and the longest loss across trees.
        assert_eq!((get(2).n, get(2).proven, get(2).dist), (15, Some(2), 2));
        assert_eq!((get(3).proven, get(3).dist), (Some(0), 7));
        assert_eq!(get(4).n, 70);
        assert_eq!(
            pick_move(&merge_stats(&a.stats(), &b.stats()), 0., 0., 64),
            2
        );
        // One tree's statistics choose like the tree itself.
        for (t, unit) in [(0., 0.), (1., 0.3), (1., 0.95)] {
            assert_eq!(pick_move(&a.stats(), t, unit, 64), a.sample_move(t, unit));
        }
    }

    #[test]
    fn sampling_follows_temperature_and_skips_proven_losses() {
        let s = tree(&[(1, 90, None, 0), (2, 10, None, 0), (3, 500, Some(0), 1)]);
        for unit in [0., 0.5, 0.99] {
            assert_eq!(s.sample_move(0., unit), 1);
        }
        // Temperature 1 weighs 90 : 10.
        assert_eq!(s.sample_move(1., 0.89), 1);
        assert_eq!(s.sample_move(1., 0.91), 2);
        // A tiny temperature behaves like the best move.
        assert_eq!(s.sample_move(1e-6, 0.99), 1);
        // A proven win outranks sampling, and the quickest one is taken.
        let s = tree(&[(1, 900, None, 0), (2, 10, Some(2), 6), (3, 5, Some(2), 2)]);
        assert_eq!(s.sample_move(2., 0.), 3);
        // When every move loses, sampling still delays the loss.
        let s = tree(&[(1, 900, Some(0), 1), (2, 10, Some(0), 5)]);
        assert_eq!(s.sample_move(2., 0.), 2);
    }

    #[test]
    fn solver_proves_a_loss_against_an_open_four() {
        let mut board = [0; 225];
        place(&[108, 109, 110, 111], 1, &mut board);
        place(&[0, 14, 210], 2, &mut board);
        let mut s = Search::new(&board, 2, 3, 2f64.sqrt(), None).unwrap();
        for _ in 0..50 {
            if s.solved() >= 0 {
                break;
            }
            run_rounds(&mut s, 1, 8);
        }
        assert_eq!(s.solved(), 0);
    }

    // Parse "D10"-style coordinates: column letter from A, row number from 1.
    fn points(names: &str) -> Vec<usize> {
        names
            .split_whitespace()
            .map(|name| {
                let x = (name.as_bytes()[0] - b'A') as usize;
                let y: usize = name[1..].parse().unwrap();
                (y - 1) * 15 + x
            })
            .collect()
    }

    #[test]
    fn solver_proves_a_four_three_from_play() {
        // White D10 made a four (B12) and a split three D10 E11 _ G13.
        let mut board = [0; 225];
        let black = "G3 I3 K3 I4 M4 H5 J5 K5 L5 O5 F6 H6 J6 K6 E7 G7 K7 G8 H8 I8 K8 \
                     D9 H9 J9 L9 M9 E10 F10 G10 I10 J10 B11 G11 I11 K11 M11 E12 H12 L12 \
                     F13 J13 G14 I14";
        let white = "G2 J3 G4 K4 L4 G5 I5 M5 I6 M6 N6 F7 H7 I7 J7 L7 M7 F8 J8 L8 M8 \
                     E9 F9 G9 I9 K9 D10 H10 L10 C11 D11 E11 F11 H11 L11 G12 I12 J12 K12 \
                     E13 G13 H13 I13";
        place(&points(black), 1, &mut board);
        place(&points(white), 2, &mut board);
        let mut s = Search::new(&board, 1, 5, 2f64.sqrt(), None).unwrap();
        assert_eq!(s.direct(), points("B12")[0] as i32);
        let mut rounds = 0;
        while s.solved() < 0 && rounds < 200 {
            run_rounds(&mut s, 1, 64);
            rounds += 1;
        }
        println!(
            "solved {} after {rounds} rounds; B12 outcomes {:?}",
            s.solved(),
            s.move_outcomes(points("B12")[0])
        );
        assert_eq!(s.solved(), 0);
    }

    #[test]
    fn priors_follow_candidate_order_and_sum_to_one() {
        let board = middle_game();
        let s = Search::new(&board, 2, 1, 1., None).unwrap();
        let priors = &s.nodes[0].priors;
        assert_eq!(priors.len(), s.nodes[0].choices.as_ref().unwrap().len());
        // Candidates are sorted by score, so priors never increase.
        assert!(priors.windows(2).all(|w| w[0] >= w[1]));
        let total: u32 = priors.iter().map(|&p| u32::from(p)).sum();
        assert!((65535 - 200..=65535 + 200).contains(&total), "{total}");
        assert!(priors[0] > priors[priors.len() - 1]);
    }

    #[test]
    fn puct_search_proves_the_open_four_win() {
        let mut board = [0; 225];
        place(&[109, 110, 111], 1, &mut board);
        place(&[0, 14, 210], 2, &mut board);
        let mut s = Search::new(&board, 1, 3, 2f64.sqrt(), None).unwrap();
        s.set_puct(0.5, 1.5);
        for _ in 0..50 {
            if s.solved() >= 0 {
                break;
            }
            run_rounds(&mut s, 1, 8);
        }
        assert_eq!(s.solved(), 2);
        assert!([108, 112].contains(&s.best_move()));
    }

    #[test]
    fn open_three_defenses_form_a_complete_set() {
        let mut board = [0; 225];
        place(&points("G8 H8 I8"), 2, &mut board);
        place(&points("A1 O1 A15"), 1, &mut board);
        let (mut choices, forced) = candidates(&board, 1, 0);
        choices.sort();
        // Only the two ends stop an open four; the outer points leave one.
        assert_eq!(choices, points("F8 J8"));
        assert!(forced);
        // Exact-five rules keep the set but do not use it for proofs.
        assert!(!candidates(&board, 1, 1).1);
    }

    #[test]
    fn proves_a_lost_record_position_and_delays_the_loss() {
        // Moves 1-67 of a played game; white faces black's B14 open four.
        let moves = [
            112, 98, 99, 113, 128, 114, 130, 115, 116, 144, 129, 127, 143, 155, 141, 139, 171, 131,
            157, 185, 170, 169, 153, 124, 154, 174, 159, 126, 125, 109, 94, 158, 173, 189, 190,
            186, 187, 201, 217, 200, 172, 202, 203, 160, 188, 176, 192, 110, 142, 184, 168, 108,
            107, 76, 92, 96, 138, 183, 182, 82, 68, 66, 50, 81, 111, 199, 198,
        ];
        let mut board = [0; 225];
        for (k, &i) in moves.iter().enumerate() {
            board[i] = 1 + (k % 2) as u8;
        }
        let mut s = Search::new(&board, 2, 7, 2f64.sqrt(), None).unwrap();
        assert!(!s.nodes[0]
            .choices
            .as_ref()
            .unwrap()
            .contains(&(points("C9")[0] as u8)));
        // The proof needs rounds rather than simulations, so keep rollouts cheap.
        while s.solved() < 0 && s.rounds < 60000 {
            run_rounds(&mut s, 1, 1);
        }
        println!(
            "record position proven after {} rounds, {} nodes (VCF in nodes)",
            s.rounds,
            s.nodes.len()
        );
        assert_eq!(s.solved(), 0);
        // Blocks lose in five plies; counter-fours hold out for nine.
        let best = s.best_move() as usize;
        assert!(points("G3 G4 C14 B15").contains(&best));
        let longest = s.root_children().map(|c| c.1.dist).max().unwrap();
        assert_eq!(
            s.root_children().find(|c| c.0 == best).unwrap().1.dist,
            longest
        );
    }

    #[test]
    fn solver_never_proves_a_loss_from_a_pruned_candidate_set() {
        let mut board = [0; 225];
        board[112] = 1;
        let mut s = Search::new(&board, 2, 9, 2f64.sqrt(), None).unwrap();
        run_rounds(&mut s, 200, 8);
        for node in &s.nodes {
            if !node.forced && node.proven.is_some() {
                assert!(
                    node.choices.is_none()
                        || node
                            .children
                            .iter()
                            .any(|&c| s.nodes[c].proven == node.proven)
                );
            }
        }
    }

    #[test]
    #[ignore = "benchmark; run with --release --ignored --nocapture"]
    fn cpu_rollout_benchmark() {
        let mut board = [0; 225];
        place(&[112, 96, 128, 98], 1, &mut board);
        place(&[113, 97, 126], 2, &mut board);
        for rule in 0..=2 {
            let mut s = Search::new(&board, 2, 42, 2f64.sqrt(), Some(rule)).unwrap();
            assert_eq!(s.select(), -1);
            let mut times = Vec::new();
            let mut tally = 0;
            for _ in 0..15 {
                let start = std::time::Instant::now();
                tally = s.cpu_batch(1024);
                times.push(start.elapsed());
            }
            times.sort();
            println!(
                "rule {rule}: 1024 rollouts median {:?} (tally {tally})",
                times[7]
            );
        }
    }

    // A middle-game position from the five-free two-colour pattern.
    fn middle_game() -> Board {
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
        board
    }

    // VCF and VCT timing on the positions of a played game, for both players.
    #[test]
    #[ignore = "benchmark; run with --release --ignored --nocapture"]
    fn threat_search_benchmark() {
        let moves = [
            112, 98, 99, 113, 128, 114, 130, 115, 116, 144, 129, 127, 143, 155, 141, 139, 171, 131,
            157, 185, 170, 169, 153, 124, 154, 174, 159, 126, 125, 109, 94, 158, 173, 189, 190,
            186, 187, 201, 217, 200, 172, 202, 203, 160, 188, 176, 192, 110, 142, 184, 168, 108,
            107, 76, 92, 96, 138, 183, 182, 82, 68, 66, 50, 81, 111, 199, 198,
        ];
        for limit in [1000, 5000, 20000] {
            let mut board = [0; 225];
            let (mut vcf_time, mut vct_time, mut vcfs, mut vcts) =
                (std::time::Duration::ZERO, std::time::Duration::ZERO, 0, 0);
            let mut slowest = std::time::Duration::ZERO;
            for (k, &i) in moves.iter().enumerate() {
                board[i] = 1 + (k % 2) as u8;
                for player in [1, 2] {
                    let t = std::time::Instant::now();
                    vcfs += u32::from(vcf::vcf(&board, player, 0, limit).is_some());
                    vcf_time += t.elapsed();
                    let t = std::time::Instant::now();
                    vcts += u32::from(vct::vct(&board, player, 0, limit).is_some());
                    let spent = t.elapsed();
                    vct_time += spent;
                    slowest = slowest.max(spent);
                }
            }
            let n = 2 * moves.len() as u32;
            println!(
                "limit {limit:5}: VCF {:?} avg ({vcfs} found), VCT {:?} avg, slowest {slowest:?} ({vcts} found) over {n} positions",
                vcf_time / n,
                vct_time / n
            );
        }
    }

    #[test]
    #[ignore = "benchmark; run with --release --ignored --nocapture"]
    fn selection_benchmark() {
        let board = middle_game();
        for rule in 0..=2 {
            let start = std::time::Instant::now();
            let mut calls = 0;
            while start.elapsed().as_millis() < 300 {
                std::hint::black_box(candidates(&board, 2, rule));
                calls += 1;
            }
            let per_call = start.elapsed() / calls;
            let mut s = Search::new(&board, 2, 42, 2f64.sqrt(), Some(rule)).unwrap();
            LEGACY_VCF.with(|flag| flag.set(std::env::var("NO_VCF").is_ok()));
            // Measure selection itself, even where the root threat search proves the root.
            s.vcf_move = None;
            s.nodes[0].proven = None;
            // Fake tallies keep the tree shape varied without paying for rollouts.
            let mut rng = 1u32;
            let start = std::time::Instant::now();
            while s.rounds < 2000 && s.solved() < 0 {
                let leaves = s.select_batch(64, 64);
                let tallies: Vec<u32> = (0..leaves)
                    .map(|_| {
                        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                        (rng >> 16) % 129
                    })
                    .collect();
                s.apply_pending(&tallies, 64);
            }
            let selections = start.elapsed();
            let start = std::time::Instant::now();
            let found = vcf::vcf(&board, 2, rule, ROOT_VCF_NODES);
            println!(
                "rule {rule}: candidates {per_call:?} per call; 2000 selections {selections:?}; {} nodes; root VCF {:?} ({found:?})",
                s.nodes.len(),
                start.elapsed()
            );
        }
    }

    // Play `games` games from short random openings, swapping colours on each opening.
    // Returns (wins, draws, losses) for engine `a`.
    #[derive(Clone, Copy)]
    struct Engine {
        solver: bool,
        temperature: f64,
        rounds: u32,
        legacy_defense: bool,
        legacy_breadth: bool,
        legacy_transpositions: bool,
        policy: bool,
        // Continue the previous move's tree after the opponent replies.
        reuse: bool,
        // Leaves selected per batch with virtual loss, as one GPU dispatch would.
        leaves: u32,
        // Batches in flight at once, as pipelined GPU dispatches would be.
        depth: usize,
        // Independent trees whose root statistics are merged.
        trees: u32,
        // PUCT constant (0 for UCT) and prior power.
        puct: f64,
        // VCF searches at the root and in expanded nodes.
        vcf: bool,
        // A VCT search at the root.
        vct: bool,
        prior_power: f64,
    }
    const ENGINE: Engine = Engine {
        solver: true,
        temperature: 0.,
        rounds: 200,
        legacy_defense: false,
        legacy_breadth: false,
        legacy_transpositions: false,
        policy: false,
        reuse: false,
        leaves: 1,
        depth: 1,
        trees: 1,
        puct: 0.5,
        prior_power: 1.5,
        vcf: true,
        vct: true,
    };

    thread_local! {
        // Seconds and moves per engine (a, b) over the self-play games.
        static MOVE_TIME: std::cell::Cell<[(f64, u32); 2]> =
            const { std::cell::Cell::new([(0., 0); 2]) };
    }
    // Average milliseconds per move of engines a and b so far.
    fn move_ms() -> [f64; 2] {
        MOVE_TIME.with(|t| t.get().map(|(secs, n)| 1000. * secs / f64::from(n.max(1))))
    }
    fn self_play(a: Engine, b: Engine, games: u32) -> (u32, u32, u32) {
        let mut result = (0, 0, 0);
        for game in 0..games {
            let mut rng = 0x9e37_79b9u32 ^ (game / 2).wrapping_mul(0x85eb_ca6b);
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                rng
            };
            let mut board = [0; 225];
            let mut player = 1;
            // Three opening stones within two squares of the centre.
            while board.iter().filter(|&&v| v != 0).count() < 3 {
                let i = (5 + next() % 5) as usize * 15 + (5 + next() % 5) as usize;
                if board[i] == 0 {
                    board[i] = player;
                    player = 3 - player;
                }
            }
            let a_plays = if game % 2 == 0 { 1 } else { 2 };
            let mut kept: [Option<Search>; 2] = [None, None];
            let winner = loop {
                let engine = if player == a_plays { a } else { b };
                LEGACY_DEFENSE.with(|flag| flag.set(engine.legacy_defense));
                LEGACY_BREADTH.with(|flag| flag.set(engine.legacy_breadth));
                LEGACY_TRANSPOSITIONS.with(|flag| flag.set(engine.legacy_transpositions));
                LEGACY_VCF.with(|flag| flag.set(!engine.vcf));
                LEGACY_VCT.with(|flag| flag.set(!engine.vct));
                let started = std::time::Instant::now();
                let seed = next();
                let slot = &mut kept[usize::from(player - 1)];
                let mut s = match slot.take() {
                    Some(mut s) if engine.reuse => {
                        if s.reroot(&board) {
                            s
                        } else {
                            Search::new(&board, player, seed, 2f64.sqrt(), None).unwrap()
                        }
                    }
                    _ => Search::new(&board, player, seed, 2f64.sqrt(), None).unwrap(),
                };
                s.solver = engine.solver;
                s.policy = engine.policy;
                s.set_puct(engine.puct, engine.prior_power);
                let mut point = s.direct();
                if point < 0 && engine.trees > 1 {
                    // Root parallelization: each tree searches alone, then the
                    // root statistics are merged.
                    let mut merged = Vec::new();
                    for _ in 0..engine.trees {
                        let mut t = Search::new(&board, player, next(), 2f64.sqrt(), None).unwrap();
                        while t.rounds < engine.rounds && t.solved() < 0 {
                            t.select_batch(engine.leaves.min(engine.rounds - t.rounds), 64);
                            t.cpu_pending(64);
                        }
                        merged = merge_stats(&merged, &t.stats());
                    }
                    point = pick_move(
                        &merged,
                        engine.temperature,
                        f64::from(next()) / 2f64.powi(32),
                        64,
                    );
                } else if point < 0 {
                    let start = s.rounds;
                    while s.rounds - start < engine.rounds && s.solved() < 0 {
                        let room = engine.rounds - (s.rounds - start) - s.in_flight();
                        let limit = engine.leaves.min(room);
                        if limit > 0 {
                            s.select_batch(limit, 64);
                            s.queue_pending();
                        }
                        if s.queued.len() >= engine.depth || limit == 0 {
                            s.cpu_finish(64);
                        }
                    }
                    s.cpu_pending(64);
                    point = s.sample_move(engine.temperature, f64::from(next()) / 2f64.powi(32));
                }
                if engine.reuse && s.focus(point as usize) {
                    *slot = Some(s);
                }
                let side = usize::from(player != a_plays);
                MOVE_TIME.with(|t| {
                    let mut t = t.get();
                    t[side].0 += started.elapsed().as_secs_f64();
                    t[side].1 += 1;
                    MOVE_TIME.set(t);
                });
                board[point as usize] = player;
                if won(&board, point as usize, player, 0) {
                    break player;
                }
                if !board.contains(&0) {
                    break 0;
                }
                player = 3 - player;
            };
            match winner {
                0 => result.1 += 1,
                w if w == a_plays => result.0 += 1,
                _ => result.2 += 1,
            }
        }
        result
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_and_without_solver() {
        let plain = Engine {
            solver: false,
            ..ENGINE
        };
        let (w, d, l) = self_play(ENGINE, plain, 100);
        println!("solver vs plain MCTS: {w} W / {d} D / {l} L");
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_against_legacy_defense() {
        let legacy = Engine {
            legacy_defense: true,
            ..ENGINE
        };
        let (w, d, l) = self_play(ENGINE, legacy, 100);
        println!("complete defense vs legacy filter: {w} W / {d} D / {l} L");
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_against_legacy_breadth() {
        let rounds = std::env::var("ROUNDS").map_or(200, |v| v.parse().unwrap());
        let engine = Engine { rounds, ..ENGINE };
        let legacy = Engine {
            legacy_breadth: true,
            ..engine
        };
        let (w, d, l) = self_play(engine, legacy, 100);
        println!("{rounds} rounds, UCT breadth vs 32 candidates: {w} W / {d} D / {l} L");
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_transpositions() {
        let rounds = std::env::var("ROUNDS").map_or(200, |v| v.parse().unwrap());
        let engine = Engine { rounds, ..ENGINE };
        let tree = Engine {
            legacy_transpositions: true,
            ..engine
        };
        let (w, d, l) = self_play(engine, tree, 100);
        println!("{rounds} rounds, transpositions vs tree: {w} W / {d} D / {l} L");
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_rollout_policy() {
        // RANDOM_ROUNDS gives the faster random rollouts a different budget.
        let var = |name, default| std::env::var(name).map_or(default, |v| v.parse().unwrap());
        let rounds = var("ROUNDS", 200);
        let policy = Engine {
            rounds,
            policy: true,
            ..ENGINE
        };
        let random = Engine {
            rounds: var("RANDOM_ROUNDS", rounds),
            ..ENGINE
        };
        let (w, d, l) = self_play(policy, random, 100);
        println!(
            "win-and-block rollouts at {rounds} rounds vs random at {} rounds: {w} W / {d} D / {l} L",
            random.rounds
        );
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_tree_reuse() {
        let rounds = std::env::var("ROUNDS").map_or(200, |v| v.parse().unwrap());
        let fresh = Engine { rounds, ..ENGINE };
        let reuse = Engine {
            reuse: true,
            ..fresh
        };
        let (w, d, l) = self_play(reuse, fresh, 100);
        println!("{rounds} rounds, reused tree vs fresh tree: {w} W / {d} D / {l} L");
    }

    // TREES independent trees of ROUNDS each against one tree of OTHER_ROUNDS.
    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_root_parallel_trees() {
        let var = |name, default| std::env::var(name).map_or(default, |v| v.parse().unwrap());
        let rounds = var("ROUNDS", 200);
        let parallel = Engine {
            rounds,
            trees: var("TREES", 4),
            ..ENGINE
        };
        let single = Engine {
            rounds: var("OTHER_ROUNDS", rounds),
            ..ENGINE
        };
        let (w, d, l) = self_play(parallel, single, 100);
        println!(
            "{} trees of {rounds} rounds vs one tree of {} rounds: {w} W / {d} D / {l} L",
            parallel.trees, single.rounds
        );
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_vcf() {
        let rounds = std::env::var("ROUNDS").map_or(200, |v| v.parse().unwrap());
        let with = Engine { rounds, ..ENGINE };
        // VCT=1 compares the root VCT against VCF alone; otherwise all against none.
        let vct_only = std::env::var("VCT").is_ok();
        let without = if vct_only {
            Engine { vct: false, ..with }
        } else {
            Engine {
                vcf: false,
                vct: false,
                ..with
            }
        };
        // OTHER_ROUNDS gives the cheaper engine more rounds, for equal time.
        let without = Engine {
            rounds: std::env::var("OTHER_ROUNDS").map_or(rounds, |v| v.parse().unwrap()),
            ..without
        };
        let (w, d, l) = self_play(with, without, 100);
        let name = if vct_only {
            "root VCT vs VCF only"
        } else {
            "VCF and VCT vs none"
        };
        let [ta, tb] = move_ms();
        println!(
            "{rounds} vs {} rounds, {name}: {w} W / {d} D / {l} L; {ta:.1} vs {tb:.1} ms per move",
            without.rounds
        );
    }

    // PUCT with constant PUCT and prior power POWER against UCT.
    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_puct() {
        let var = |name, default| std::env::var(name).map_or(default, |v| v.parse().unwrap());
        let rounds = var("ROUNDS", 200.) as u32;
        let leaves = var("LEAVES", 1.) as u32;
        let depth = var("DEPTH", 1.) as usize;
        let puct = Engine {
            rounds,
            puct: var("PUCT", 0.5),
            prior_power: var("POWER", 1.5),
            leaves,
            depth,
            ..ENGINE
        };
        let uct = Engine {
            rounds,
            leaves,
            depth,
            puct: 0.,
            ..ENGINE
        };
        let (w, d, l) = self_play(puct, uct, 100);
        println!(
            "PUCT {} power {} vs UCT, {rounds} rounds: {w} W / {d} D / {l} L",
            puct.puct, puct.prior_power
        );
    }

    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_with_parallel_leaves() {
        let var =
            |name, default| std::env::var(name).map_or(default, |v: String| v.parse().unwrap());
        let rounds = var("ROUNDS", 1000);
        let parallel = Engine {
            rounds,
            leaves: var("LEAVES", 64),
            depth: var("DEPTH", 1) as usize,
            ..ENGINE
        };
        // OTHER_* configure the opponent, sequential by default.
        let other = Engine {
            rounds: var("OTHER_ROUNDS", rounds),
            leaves: var("OTHER_LEAVES", 1),
            ..ENGINE
        };
        let (w, d, l) = self_play(parallel, other, 100);
        println!(
            "{} leaves x {} in flight at {rounds} rounds vs {} leaves at {} rounds: {w} W / {d} D / {l} L",
            parallel.leaves, parallel.depth, other.leaves, other.rounds
        );
    }

    // TEMPERATURE and ROUNDS pick the weaker engine; it plays the default engine.
    #[test]
    #[ignore = "strength match; run with --release --ignored --nocapture"]
    fn strength_by_difficulty() {
        let var =
            |name, default| std::env::var(name).map_or(default, |v: String| v.parse().unwrap());
        let weak = Engine {
            temperature: var("TEMPERATURE", 1.),
            rounds: var("ROUNDS", 200.) as u32,
            ..ENGINE
        };
        let (w, d, l) = self_play(weak, ENGINE, 100);
        println!(
            "temperature {} rounds {} vs default: {w} W / {d} D / {l} L",
            weak.temperature, weak.rounds
        );
    }

    #[test]
    fn returned_moves_are_legal() {
        for rule in 0..=2 {
            for seed in 1..=3 {
                let mut board = [0; 225];
                let mut player = 1;
                for _ in 0..40 {
                    if rule == 2 && player == 1 && !black_can_move(&board) {
                        break;
                    }
                    let mut s = Search::new(&board, player, seed, 2f64.sqrt(), Some(rule)).unwrap();
                    let mut point = s.direct();
                    if point < 0 {
                        run_rounds(&mut s, 6, 4);
                        point = s.best_move();
                    }
                    let point = point as usize;
                    assert!(
                        legal_move(&board, point, player, rule),
                        "rule {rule}, seed {seed}"
                    );
                    board[point] = player;
                    if won(&board, point, player, rule) || !board.contains(&0) {
                        break;
                    }
                    player = 3 - player;
                }
            }
        }
    }

    #[test]
    fn move_orders_share_one_node() {
        let mut board = [0; 225];
        board[112] = 1;
        let mut s = Search::new(&board, 2, 5, 2f64.sqrt(), None).unwrap();
        run_rounds(&mut s, 3000, 1);
        // Every node has one key, and the edges outnumber the nodes once two
        // move orders meet.
        assert_eq!(s.table.len(), s.nodes.len());
        let edges: usize = s.nodes.iter().map(|n| n.children.len()).sum();
        assert!(
            edges > s.nodes.len() - 1,
            "{edges} edges, {} nodes",
            s.nodes.len()
        );
        // Each edge leads to the position its move makes.
        let mut stack = vec![(0, board, 2u8)];
        let mut seen = vec![false; s.nodes.len()];
        while let Some((id, b, turn)) = stack.pop() {
            assert_eq!(s.table[&zobrist(&b)], id);
            if std::mem::replace(&mut seen[id], true) {
                continue;
            }
            for (i, &child) in s.edges(id).map(|(i, _)| i).zip(&s.nodes[id].children) {
                let mut after = b;
                after[i] = turn;
                stack.push((child, after, 3 - turn));
            }
        }
    }

    // Check that each edge leads to the position its move makes.
    fn assert_consistent(s: &Search) {
        assert_eq!(s.table.len(), s.nodes.len());
        let mut stack = vec![(0, s.board, s.root)];
        let mut seen = vec![false; s.nodes.len()];
        while let Some((id, b, turn)) = stack.pop() {
            assert_eq!(s.table[&zobrist(&b)], id);
            if std::mem::replace(&mut seen[id], true) {
                continue;
            }
            for (i, &child) in s.edges(id).map(|(i, _)| i).zip(&s.nodes[id].children) {
                let mut after = b;
                after[i] = turn;
                stack.push((child, after, 3 - turn));
            }
        }
        assert!(seen.iter().all(|&v| v), "unreachable nodes after reroot");
    }

    #[test]
    fn reroot_keeps_the_subtree_of_the_played_moves() {
        let mut board = [0; 225];
        place(&[112, 96], 1, &mut board);
        place(&[113], 2, &mut board);
        let mut s = Search::new(&board, 2, 5, 2f64.sqrt(), None).unwrap();
        run_rounds(&mut s, 300, 8);
        let mine = s.best_move() as usize;
        assert!(s.focus(mine));
        let others: Vec<u32> = s
            .root_children()
            .filter(|c| c.0 != mine)
            .map(|c| c.1.n)
            .collect();
        run_rounds(&mut s, 200, 8);
        // Pondering stays below the played move.
        let after: Vec<u32> = s
            .root_children()
            .filter(|c| c.0 != mine)
            .map(|c| c.1.n)
            .collect();
        assert_eq!(others, after);
        let k = s.nodes[0]
            .choices
            .as_ref()
            .unwrap()
            .iter()
            .position(|&i| usize::from(i) == mine)
            .unwrap();
        let played = s.nodes[0].children[k];
        let (reply, node) = s
            .edges(played)
            .max_by_key(|c| c.1.n)
            .map(|(i, n)| (i, (n.n, n.wins)))
            .unwrap();
        assert!(s.nodes[played].n > node.0);
        let mut next = board;
        next[mine] = 2;
        next[reply] = 1;
        // Wrong positions are refused and leave the search intact.
        let mut wrong = next;
        wrong[reply] = 0;
        assert!(!s.reroot(&wrong));
        assert!(s.reroot(&next));
        assert_eq!((s.nodes[0].n, s.simulation_wins()), node);
        assert_consistent(&s);
        // The reused root keeps searching like any other.
        let total = s.nodes[0].n + run_rounds(&mut s, 50, 8);
        assert_eq!(s.nodes[0].n, total);
        assert_consistent(&s);
        assert!(s.best_move() >= 0);
    }

    #[test]
    fn focus_stats_list_the_replies_to_the_played_move() {
        let mut board = [0; 225];
        board[112] = 1;
        let mut s = Search::new(&board, 2, 5, 0.5, None).unwrap();
        run_rounds(&mut s, 200, 8);
        assert!(s.focus_stats().is_empty());
        let mine = s.best_move() as usize;
        assert!(s.focus(mine));
        run_rounds(&mut s, 200, 8);
        let replies = s.focus_stats();
        assert!(!replies.is_empty());
        let k = s.nodes[0]
            .choices
            .as_ref()
            .unwrap()
            .iter()
            .position(|&i| usize::from(i) == mine)
            .unwrap();
        let played = s.nodes[0].children[k];
        let total: u32 = replies.chunks(8).map(|w| w[1]).sum();
        assert!(total <= s.nodes[played].n);
        assert!(replies
            .chunks(8)
            .all(|w| board[w[0] as usize] == 0 && w[0] as usize != mine));
    }

    #[test]
    fn reroot_keeps_proofs() {
        without_vcf();
        let mut board = [0; 225];
        place(&[109, 110, 111], 1, &mut board);
        place(&[0, 14, 210], 2, &mut board);
        let mut s = Search::new(&board, 1, 3, 2f64.sqrt(), None).unwrap();
        while s.solved() < 0 {
            run_rounds(&mut s, 1, 8);
        }
        let mine = s.best_move() as usize;
        assert!(s.focus(mine));
        assert_eq!(s.solved(), 2);
        let mut next = board;
        next[mine] = 1;
        // White blocks one end of the open four.
        let reply = if mine == 108 { 112 } else { 108 };
        next[reply] = 2;
        assert!(s.reroot(&next));
        assert_eq!(s.solved(), 2);
        assert_eq!(s.nodes[0].dist, 1);
        assert_consistent(&s);
    }

    #[test]
    fn cancelled_expansions_leave_no_trace() {
        let mut board = [0; 225];
        board[112] = 1;
        let mut s = Search::new(&board, 2, 5, 2f64.sqrt(), None).unwrap();
        run_rounds(&mut s, 100, 8);
        let (nodes, keys) = (s.nodes.len(), s.table.len());
        assert_eq!(s.select(), -1);
        s.cancel_selection();
        assert_eq!((s.nodes.len(), s.table.len()), (nodes, keys));
        let edges: usize = s.nodes.iter().map(|n| n.children.len()).sum();
        run_rounds(&mut s, 1, 8);
        let after: usize = s.nodes.iter().map(|n| n.children.len()).sum();
        assert!(after <= edges + 1);
    }

    #[test]
    fn window_counts_match_the_line_scan() {
        let mut rng = 7u32;
        let mut next = |n: u32| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng % n
        };
        for game in 0..300 {
            let mut board = [0; 225];
            // Some boards block the outer rows like the smaller board sizes.
            if game % 3 == 0 {
                for (i, cell) in board.iter_mut().enumerate() {
                    if i / 15 >= 13 || i % 15 >= 13 {
                        *cell = 3;
                    }
                }
            }
            for _ in 0..next(90) {
                let i = next(225) as usize;
                if board[i] == 0 {
                    board[i] = 1 + next(2) as u8;
                }
            }
            let windows = Windows::new(&board);
            for rule in 0..=2 {
                for i in (0..225).filter(|&i| board[i] == 0) {
                    for p in [1, 2] {
                        // `won` is only meaningful when no five is on the board yet.
                        let five_on_board = (0..225)
                            .any(|j| matches!(board[j], 1 | 2) && won(&board, j, board[j], 0));
                        if !five_on_board {
                            assert_eq!(windows.wins(&board, i, p, rule), won(&board, i, p, rule));
                            let old = threat(&board, i, p, rule);
                            let new = windows.threat(&board, i, p, rule);
                            assert_eq!(new.value, old.value);
                            assert_eq!(new.four, old.winning_replies.first().copied());
                            assert_eq!(new.fork, old.winning_replies.len() >= 2);
                            let scan = windows.scan(i, p);
                            let other = threat(&board, i, 3 - p, rule);
                            assert_eq!(scan.value, old.value + other.value);
                            assert_eq!(scan.five, [old.value >= 10000, other.value >= 10000]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn excludes_counterthreat_when_forced_reply_creates_an_enemy_fork() {
        let mut board = [0; 225];
        for i in [109, 110, 111] {
            board[i] = 1;
        }
        for i in [107, 82, 97, 127] {
            board[i] = 2;
        }
        let candidates = moves(&board, 1, 0);
        assert!(candidates.contains(&112));
        assert!(!candidates.contains(&108));
    }
}
