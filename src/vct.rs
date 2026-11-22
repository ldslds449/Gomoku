// Victory by continuous threats under the five-or-more rule, searched with
// depth-first proof-number search (df-pn). The attacker plays fours and threes:
// moves after which it has a five point or a fork point (a point giving two
// winning replies). Against a four the defender must block; against a three
// its complete defense set is every counter-four and every move that leaves
// the attacker no fork point. Any other move loses to the fork at once, so a
// proof over these sets is a real proof.
use super::{windows::Windows, Board, ZOBRIST};
use std::collections::HashMap;

const INF: u32 = u32::MAX / 4;
// Ply limits tried in turn, so shorter wins are found first.
const PLY_LIMITS: [u16; 5] = [5, 9, 13, 17, 21];

#[derive(Clone, Copy)]
struct Entry {
    pn: u32,
    dn: u32,
    // Plies to the five once proven.
    dist: u16,
}

pub struct Vct {
    board: Board,
    windows: Windows,
    key: u64,
    attacker: u8,
    nodes: u32,
    limit: u32,
    max_plies: u16,
    table: HashMap<u64, Entry>,
}

// The first attacker move of a VCT found within `limit` nodes, and the plies
// until the five along the proof. Only the five-or-more rule is supported.
pub fn vct(board: &Board, attacker: u8, rule: u8, limit: u32) -> Option<(usize, u16)> {
    vct_within(board, attacker, rule, limit, &PLY_LIMITS)
}
fn vct_within(
    board: &Board,
    attacker: u8,
    rule: u8,
    limit: u32,
    ply_limits: &[u16],
) -> Option<(usize, u16)> {
    if rule != 0 {
        return None;
    }
    let mut search = Vct {
        board: *board,
        windows: Windows::new(board),
        key: super::zobrist(board),
        attacker,
        nodes: 0,
        limit,
        max_plies: 0,
        table: HashMap::new(),
    };
    // Deepen the ply limit, sharing the node budget; a limit's disproofs do
    // not hold under the next one, so each starts a fresh table.
    let proven = ply_limits.iter().any(|&plies| {
        if search.nodes >= search.limit {
            return false;
        }
        search.max_plies = plies;
        search.table.clear();
        search.mid(INF - 1, INF - 1, 0).0 == 0
    });
    if !proven {
        return None;
    }
    // The proven child with the fewest plies.
    search
        .attacks()
        .into_iter()
        .filter_map(|m| {
            search.place(m, attacker);
            let entry = search.table.get(&search.key).copied();
            let immediate = search.attacker_fives() >= 2;
            search.remove(m, attacker);
            if immediate {
                return Some((m, 3));
            }
            entry
                .filter(|e| e.pn == 0)
                .map(|e| (m, e.dist.saturating_add(1)))
        })
        .min_by_key(|&(_, dist)| dist)
}

impl Vct {
    fn place(&mut self, i: usize, player: u8) {
        self.board[i] = player;
        self.windows.place(i, player);
        self.key ^= ZOBRIST[i][usize::from(player - 1)];
    }
    fn remove(&mut self, i: usize, player: u8) {
        self.board[i] = 0;
        self.windows.remove(i, player);
        self.key ^= ZOBRIST[i][usize::from(player - 1)];
    }
    fn attacker_fives(&self) -> usize {
        self.windows
            .five_points(&self.board, self.attacker, 0)
            .len()
    }
    // Points where `player` makes a fork: two distinct winning replies.
    fn forks(&self, player: u8) -> Vec<usize> {
        self.windows
            .four_points(&self.board, player)
            .into_iter()
            .filter(|&i| self.windows.threat(&self.board, i, player, 0).fork)
            .collect()
    }
    // Attacker moves at an OR node; none when two defender fives cannot both
    // be blocked or the forced block is no threat.
    fn attacks(&mut self) -> Vec<usize> {
        let (attacker, defender) = (self.attacker, 3 - self.attacker);
        let threats = self.windows.five_points(&self.board, defender, 0);
        let candidates = match threats.len() {
            0 => {
                // Fours, and threes: cells of open windows with two attacker
                // stones that may leave a fork point.
                let mut cells = self.windows.four_points(&self.board, attacker);
                for i in self.windows.three_points(&self.board, attacker) {
                    if !cells.contains(&i) {
                        cells.push(i);
                    }
                }
                cells
            }
            1 => threats,
            _ => return Vec::new(),
        };
        let mut moves = Vec::new();
        for m in candidates {
            self.place(m, attacker);
            let threat = self.attacker_fives() > 0 || !self.forks(attacker).is_empty();
            self.remove(m, attacker);
            if threat {
                moves.push(m);
            }
        }
        moves
    }
    // Defender moves at an AND node: the block of a four, or the complete
    // defense set against a three.
    fn defenses(&mut self) -> Vec<usize> {
        let (attacker, defender) = (self.attacker, 3 - self.attacker);
        let fives = self.windows.five_points(&self.board, attacker, 0);
        if !fives.is_empty() {
            return fives;
        }
        let forks = self.forks(attacker);
        // Only a stone on an open window through a fork point can stop it.
        let mut cells = Vec::new();
        for &p in &forks {
            for i in self.windows.cells_through(p, attacker, 3) {
                if self.board[i] == 0 && !cells.contains(&i) {
                    cells.push(i);
                }
            }
        }
        let mut moves = Vec::new();
        for d in cells {
            self.place(d, defender);
            let stops = self.forks(attacker).is_empty();
            self.remove(d, defender);
            if stops {
                moves.push(d);
            }
        }
        // Counter-fours force a reply wherever they are.
        for d in self.windows.four_points(&self.board, defender) {
            if !moves.contains(&d)
                && self
                    .windows
                    .threat(&self.board, d, defender, 0)
                    .four
                    .is_some()
            {
                moves.push(d);
            }
        }
        moves
    }
    // Multiple-iterative deepening at the current node, with the attacker to
    // move on even plies. Returns the node's proof and disproof numbers.
    fn mid(&mut self, th_pn: u32, th_dn: u32, ply: u16) -> (u32, u32) {
        let or = ply.is_multiple_of(2);
        if or && self.attacker_fives() > 0 {
            self.store(0, INF, 1);
            return (0, INF);
        }
        if self.nodes >= self.limit {
            return (1, 1);
        }
        // Too deep counts as disproven, so the search moves elsewhere.
        if ply >= self.max_plies {
            self.store(INF, 0, 0);
            return (INF, 0);
        }
        self.nodes += 1;
        let moves = if or {
            self.attacks()
        } else {
            // Two attacker fives cannot both be blocked.
            if self.attacker_fives() >= 2 {
                self.store(0, INF, 2);
                return (0, INF);
            }
            self.defenses()
        };
        if moves.is_empty() {
            // No threat to play is a disproof. No defense is a proof: any
            // move, the fork, a block, then the five.
            let (pn, dn) = if or { (INF, 0) } else { (0, INF) };
            self.store(pn, dn, if or { 0 } else { 4 });
            return (pn, dn);
        }
        let mover = if or { self.attacker } else { 3 - self.attacker };
        loop {
            // Children's numbers from the table, unknown ones as (1, 1).
            let mut children = Vec::with_capacity(moves.len());
            for &m in &moves {
                self.place(m, mover);
                let entry = self.table.get(&self.key).copied();
                self.remove(m, mover);
                children.push(entry.map_or((1, 1, 0), |e| (e.pn, e.dn, e.dist)));
            }
            let (pn, dn) = if or {
                (
                    children.iter().map(|c| c.0).min().unwrap(),
                    children
                        .iter()
                        .map(|c| c.1)
                        .fold(0, |a: u32, b| a.saturating_add(b).min(INF)),
                )
            } else {
                (
                    children
                        .iter()
                        .map(|c| c.0)
                        .fold(0, |a: u32, b| a.saturating_add(b).min(INF)),
                    children.iter().map(|c| c.1).min().unwrap(),
                )
            };
            if pn >= th_pn || dn >= th_dn || pn == 0 || dn == 0 || self.nodes >= self.limit {
                let dist = if pn == 0 {
                    // Quickest proven attack, or the longest defense.
                    let proven = children.iter().filter(|c| c.0 == 0).map(|c| c.2);
                    1 + if or { proven.min() } else { proven.max() }.unwrap_or(0)
                } else {
                    0
                };
                self.store(pn, dn, dist);
                return (pn, dn);
            }
            // Descend into the most promising child with tightened thresholds.
            let score = |c: &(u32, u32, u16)| if or { c.0 } else { c.1 };
            let mut order: Vec<usize> = (0..children.len()).collect();
            order.sort_by_key(|&k| score(&children[k]));
            let best = order[0];
            let second = order.get(1).map_or(INF, |&k| score(&children[k]));
            let (c_pn, c_dn, _) = children[best];
            let (child_th_pn, child_th_dn) = if or {
                (
                    th_pn.min(second.saturating_add(1)),
                    (th_dn - dn).saturating_add(c_dn),
                )
            } else {
                (
                    (th_pn - pn).saturating_add(c_pn),
                    th_dn.min(second.saturating_add(1)),
                )
            };
            let m = moves[best];
            self.place(m, mover);
            self.mid(child_th_pn.min(INF), child_th_dn.min(INF), ply + 1);
            self.remove(m, mover);
        }
    }
    fn store(&mut self, pn: u32, dn: u32, dist: u16) {
        self.table.insert(self.key, Entry { pn, dn, dist });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn board(black: &[usize], white: &[usize]) -> Board {
        let mut b = [0; 225];
        for &i in black {
            b[i] = 1;
        }
        for &i in white {
            b[i] = 2;
        }
        b
    }
    #[test]
    fn finds_a_three_three_without_any_four() {
        // Two open twos crossing at 112: black 110 111 on the row and 82 97 on
        // the column. 112 makes two open threes, so a fork follows either way.
        let b = board(&[110, 111, 82, 97], &[0, 14, 210, 224]);
        assert!(crate::vcf::vcf(&b, 1, 0, 10000).is_none());
        let (first, plies) = vct(&b, 1, 0, 20000).expect("a VCT");
        assert_eq!(first, 112);
        assert!(plies >= 5, "{plies}");
    }
    #[test]
    fn a_lone_two_is_no_vct() {
        let b = board(&[110, 111], &[0, 14, 210]);
        assert!(vct(&b, 1, 0, 20000).is_none());
    }
    // The defense set equals a full-board scan: every counter-four and every
    // point that leaves the attacker no fork, on random crowded boards.
    #[test]
    fn defense_sets_are_complete() {
        let mut rng = 5u32;
        let mut next = |n: u32| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng % n
        };
        let mut checked = 0;
        for _ in 0..600 {
            let mut b = [0; 225];
            let mut p = 1;
            for _ in 0..10 + next(30) {
                let i = (3 + next(9)) as usize * 15 + (3 + next(9)) as usize;
                if b[i] == 0 {
                    b[i] = p;
                    p = 3 - p;
                }
            }
            if (0..225).any(|j| matches!(b[j], 1 | 2) && crate::won(&b, j, b[j], 0)) {
                continue;
            }
            let attacker = 3 - p;
            let mut search = Vct {
                board: b,
                windows: Windows::new(&b),
                key: 0,
                attacker,
                nodes: 0,
                limit: 0,
                max_plies: 21,
                table: HashMap::new(),
            };
            // Only positions where the attacker threatens a fork, with no fives.
            if search.forks(attacker).is_empty()
                || search.attacker_fives() > 0
                || !search.windows.five_points(&b, p, 0).is_empty()
            {
                continue;
            }
            let mut set = search.defenses();
            set.sort();
            let mut full: Vec<usize> = (0..225)
                .filter(|&d| b[d] == 0)
                .filter(|&d| {
                    let four = search.windows.threat(&b, d, p, 0).four.is_some();
                    search.place(d, p);
                    let stops = search.forks(attacker).is_empty();
                    search.remove(d, p);
                    four || stops
                })
                .collect();
            full.sort();
            assert_eq!(set, full);
            checked += 1;
        }
        assert!(checked > 30, "{checked}");
    }
    // After the first move of a VCT found on a random board, the attacker
    // still wins against every defense.
    #[test]
    fn every_defense_still_loses() {
        let mut rng = 9u32;
        let mut next = |n: u32| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng % n
        };
        let mut found = 0;
        for _ in 0..300 {
            let mut b = [0; 225];
            let mut p = 1;
            for _ in 0..8 + next(24) {
                let i = (4 + next(7)) as usize * 15 + (4 + next(7)) as usize;
                if b[i] == 0 {
                    b[i] = p;
                    p = 3 - p;
                }
            }
            if (0..225).any(|j| matches!(b[j], 1 | 2) && crate::won(&b, j, b[j], 0)) {
                continue;
            }
            let attacker = p;
            // A few proofs keep the debug-build test short.
            if found == 6 {
                break;
            }
            let Some((first, plies)) = vct(&b, attacker, 0, 5000) else {
                continue;
            };
            found += 1;
            b[first] = attacker;
            let mut search = Vct {
                board: b,
                windows: Windows::new(&b),
                key: 0,
                attacker,
                nodes: 0,
                limit: 0,
                max_plies: 21,
                table: HashMap::new(),
            };
            if search.attacker_fives() >= 2 {
                continue;
            }
            for d in search.defenses() {
                let mut after = b;
                after[d] = 3 - attacker;
                let left = crate::vcf::vcf(&after, attacker, 0, 20000)
                    .or_else(|| vct(&after, attacker, 0, 50000))
                    .map(|v| v.1);
                // df-pn proofs need not be the shortest, so only the win is checked.
                assert!(left.is_some(), "{first} then {d}: no win after {plies}");
            }
        }
        assert_eq!(found, 6);
    }
}
