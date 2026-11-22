// Victory by continuous fours: the attacker plays only fours, so each
// defender reply is forced to the one winning point, until a double four or
// an unstoppable four wins. Iterative deepening finds the line with the fewest
// attacker moves first, and failed positions are remembered by key.
use super::{rules, windows::Windows, Board, ZOBRIST};
use std::collections::HashMap;

// Attacker fours tried at most, deepening one at a time.
const MAX_FOURS: u16 = 12;

pub struct Vcf {
    board: Board,
    windows: Windows,
    key: u64,
    rule: u8,
    attacker: u8,
    nodes: u32,
    limit: u32,
    // Positions (by key) that failed within at most this many more fours.
    failed: HashMap<u64, u16>,
    // Whether the last iteration stopped anywhere for want of depth.
    cut: bool,
}

// The first attacker move of the quickest VCF found within `limit` nodes, and
// the plies until the five (1 when it is already there).
pub fn vcf(board: &Board, attacker: u8, rule: u8, limit: u32) -> Option<(usize, u16)> {
    vcf_with(board, Windows::new(board), attacker, rule, limit)
}
// The same with the window counts of `board` already at hand.
pub fn vcf_with(
    board: &Board,
    windows: Windows,
    attacker: u8,
    rule: u8,
    limit: u32,
) -> Option<(usize, u16)> {
    // Without a five or a three in an open window there is no four to play.
    if windows.five_points(board, attacker, rule).is_empty()
        && windows.four_points(board, attacker).is_empty()
    {
        return None;
    }
    let mut search = Vcf {
        board: *board,
        windows,
        key: super::zobrist(board),
        rule,
        attacker,
        nodes: 0,
        limit,
        failed: HashMap::new(),
        cut: false,
    };
    for fours in 0..=MAX_FOURS {
        search.cut = false;
        if let Some(found) = search.attack(fours) {
            return Some(found);
        }
        // Nothing deeper to find once an iteration ran out of fours, not depth.
        if search.nodes >= search.limit || !search.cut {
            return None;
        }
    }
    None
}

impl Vcf {
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
    // The attacker to move wins with at most `fours` more fours: returns the
    // first move and the plies to the five.
    fn attack(&mut self, fours: u16) -> Option<(usize, u16)> {
        let (attacker, defender, rule) = (self.attacker, 3 - self.attacker, self.rule);
        if let Some(&five) = self
            .windows
            .five_points(&self.board, attacker, rule)
            .first()
        {
            return Some((five, 1));
        }
        if fours == 0 {
            // Fours remain to try only if some four exists here.
            self.cut |= !self.windows.four_points(&self.board, attacker).is_empty()
                || !self
                    .windows
                    .five_points(&self.board, 3 - attacker, rule)
                    .is_empty();
            return None;
        }
        if self.nodes >= self.limit {
            return None;
        }
        if self
            .failed
            .get(&self.key)
            .is_some_and(|&depth| depth >= fours)
        {
            return None;
        }
        self.nodes += 1;
        // A defender five must be blocked first, and the block must be a four.
        let threats = self.windows.five_points(&self.board, defender, rule);
        let moves = match threats.len() {
            0 => self.windows.four_points(&self.board, attacker),
            1 => threats,
            _ => return self.fail(fours),
        };
        for m in moves {
            if self.board[m] != 0 || !rules::legal(&self.board, m, attacker, rule) {
                continue;
            }
            let threat = self.windows.threat(&self.board, m, attacker, rule);
            let Some(reply) = threat.four else {
                continue;
            };
            // Two winning points: the defender blocks one, the attacker plays
            // the other.
            if threat.fork {
                return Some((m, 3));
            }
            self.place(m, attacker);
            // A forbidden block leaves black without a defense.
            let found = if !rules::legal(&self.board, reply, defender, rule) {
                Some(3)
            } else {
                self.place(reply, defender);
                let found = self.attack(fours - 1).map(|(_, plies)| plies + 2);
                self.remove(reply, defender);
                found
            };
            self.remove(m, attacker);
            if let Some(plies) = found {
                return Some((m, plies));
            }
        }
        self.fail(fours)
    }
    fn fail(&mut self, fours: u16) -> Option<(usize, u16)> {
        // A search cut short by the node limit proves nothing.
        if self.nodes < self.limit {
            let depth = self.failed.entry(self.key).or_insert(0);
            *depth = (*depth).max(fours);
        }
        None
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
    fn finds_an_open_four_and_an_existing_five() {
        // Black 109 110 111 (open three on row 7): 108 or 112 makes an open four.
        let b = board(&[109, 110, 111], &[0, 14, 210]);
        let (first, plies) = vcf(&b, 1, 0, 1000).unwrap();
        assert!([108, 112].contains(&first));
        assert_eq!(plies, 3);
        let b = board(&[108, 109, 110, 111], &[0, 14, 210]);
        assert_eq!(vcf(&b, 1, 0, 1000).unwrap().1, 1);
    }
    #[test]
    fn finds_a_four_four_through_forced_blocks() {
        // Two broken threes sharing 112, each closed at one end: 112 makes a
        // four on the row (reply 109) and on the column (reply 67) at once.
        let b = board(&[110, 111, 113, 82, 97, 127], &[114, 142, 0, 14, 210, 224]);
        let (first, plies) = vcf(&b, 1, 0, 1000).unwrap();
        assert_eq!(first, 112);
        assert_eq!(plies, 3);
    }
    #[test]
    fn needs_fours_not_threes() {
        // A lone open three has no VCF for the other side.
        let b = board(&[109, 110, 111], &[0, 14, 210]);
        assert!(vcf(&b, 2, 0, 1000).is_none());
        // Blocked threes give single fours that the defender stops.
        let b = board(&[109, 110, 111], &[108, 112, 210]);
        assert!(vcf(&b, 1, 0, 1000).is_none());
    }
    #[test]
    fn exact_five_rules_reject_fours_that_overline() {
        // Black 105 106 107 _ 109 110: 108 makes six, not five, under exact five.
        let b = board(&[105, 106, 107, 109, 110], &[0, 14, 210, 224, 30]);
        assert_eq!(vcf(&b, 1, 0, 1000).map(|v| v.1), Some(1));
        assert!(vcf(&b, 1, 1, 1000).is_none_or(|v| v.1 > 1));
    }
    // Replay every VCF found on random crowded boards: each first move is
    // legal, and after the forced block the attacker still wins two plies sooner.
    #[test]
    fn found_lines_replay_to_a_five() {
        use crate::won;
        let mut rng = 3u32;
        let mut next = |n: u32| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng % n
        };
        let mut found = 0;
        for game in 0..400 {
            let rule = (game % 3) as u8;
            let mut b = [0; 225];
            let mut p = 1;
            for _ in 0..20 + next(30) {
                let i = (3 + next(9)) as usize * 15 + (3 + next(9)) as usize;
                if b[i] == 0 {
                    b[i] = p;
                    p = 3 - p;
                }
            }
            if (0..225).any(|j| matches!(b[j], 1 | 2) && won(&b, j, b[j], rule)) {
                continue;
            }
            let attacker = p;
            let Some((mut first, mut plies)) = vcf(&b, attacker, rule, 20000) else {
                continue;
            };
            found += 1;
            loop {
                assert_eq!(b[first], 0);
                assert!(rules::legal(&b, first, attacker, rule), "game {game}");
                b[first] = attacker;
                if plies == 1 {
                    assert!(won(&b, first, attacker, rule), "game {game}");
                    break;
                }
                // The defender blocks a winning point if it can.
                let windows = Windows::new(&b);
                let fives = windows.five_points(&b, attacker, rule);
                assert!(!fives.is_empty(), "game {game}: no four after {first}");
                assert!(windows.five_points(&b, 3 - attacker, rule).is_empty());
                let block = fives[0];
                if !rules::legal(&b, block, 3 - attacker, rule) {
                    break;
                }
                b[block] = 3 - attacker;
                let (m, left) = vcf(&b, attacker, rule, 20000).expect("the line continues");
                assert!(left <= plies - 2, "game {game}: {left} after {plies}");
                (first, plies) = (m, left);
            }
        }
        assert!(found > 20, "{found} lines");
    }
}
