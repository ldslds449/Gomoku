// A line fits in 15 bits. Pack one verdict per possible mask into 4 KiB.
const fn build_table() -> [u32; 1024] {
    let mut table = [0; 1024];
    let mut mask = 0usize;
    while mask < 1 << 15 {
        if mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4) != 0 {
            table[mask / 32] |= 1 << (mask % 32);
        }
        mask += 1;
    }
    table
}
pub const WIN_TABLE: [u32; 1024] = build_table();
#[derive(Clone)]
pub struct Lines([[u16; 88]; 2]);
impl Lines {
    pub fn from_board(board: &super::Board) -> Self {
        let mut lines = Self([[0; 88]; 2]);
        for (i, &player) in board.iter().enumerate() {
            if matches!(player, 1 | 2) {
                lines.place(i, player);
            }
        }
        lines
    }
    // The line and the position along it for each axis through a point:
    // rows, columns, then the x - y and x + y diagonals.
    pub fn slots(i: usize) -> [(usize, usize); 4] {
        let (x, y) = (i % 15, i / 15);
        [(y, x), (15 + x, y), (30 + x + 14 - y, x), (59 + x + y, x)]
    }
    // Whether a stone at the empty point `i` would make five. No five exists
    // yet, so any five in a line after the move runs through `i`.
    pub fn wins_at(&self, i: usize, player: u8, exact: bool) -> bool {
        (0..4).any(|a| self.wins_on(i, player, exact, a))
    }
    // Whether a stone at the empty point `i` would make five along axis `a`.
    pub fn wins_on(&self, i: usize, player: u8, exact: bool, a: usize) -> bool {
        let (line, pos) = Self::slots(i)[a];
        let mask = self.0[(player - 1) as usize][line] as usize | 1 << pos;
        if exact {
            let mask = mask as u32;
            (!(mask >> pos)).trailing_zeros() + (!(mask << (31 - pos))).leading_zeros() == 6
        } else {
            WIN_TABLE[mask / 32] & (1 << (mask % 32)) != 0
        }
    }
    // Whether the line of axis `a` through `i` holds at least four of the
    // player's stones within four cells of `i`, so a winning point may be near.
    pub fn crowded(&self, i: usize, player: u8, a: usize) -> bool {
        let (line, pos) = Self::slots(i)[a];
        ((u32::from(self.0[(player - 1) as usize][line]) << 4 >> pos) & 0x1ff).count_ones() >= 4
    }
    pub fn place(&mut self, i: usize, player: u8) -> bool {
        self.place_with_rule(i, player, false)
    }
    pub fn place_with_rule(&mut self, i: usize, player: u8, exact: bool) -> bool {
        let x = i % 15;
        let y = i / 15;
        let lines = &mut self.0[(player - 1) as usize];
        let mut win = false;
        for (line, pos) in [(y, x), (15 + x, y), (30 + x + 14 - y, x), (59 + x + y, x)] {
            lines[line] |= 1 << pos;
            let mask = lines[line] as usize;
            win |= if exact {
                let mask = mask as u32;
                (!(mask >> pos)).trailing_zeros() + (!(mask << (31 - pos))).leading_zeros() == 6
            } else {
                WIN_TABLE[mask / 32] & (1 << (mask % 32)) != 0
            };
        }
        win
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_scan_at_every_move() {
        for seed in 1..=100u32 {
            let mut board = [0; 225];
            let mut lines = Lines::from_board(&board);
            let mut cells: Vec<usize> = (0..225).collect();
            let mut rng = seed;
            for turn in 0..225 {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                let i = cells.swap_remove(rng as usize % cells.len());
                let p = 1 + (turn % 2) as u8;
                board[i] = p;
                let win = lines.place(i, p);
                assert_eq!(win, crate::won(&board, i, p, 0));
                if win {
                    break;
                }
            }
        }
    }
}
