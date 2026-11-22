// Stone counts for every five-cell window on the board. Candidate scoring reads
// the twenty windows through a point instead of rescanning the lines.
use super::{rules, won, Board};

const AXES: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
pub const COUNT: usize = 572;
const NONE: u16 = u16::MAX;

// Window ids by axis and first cell, then the cells of each window.
const fn build_starts() -> ([[u16; 225]; 4], [[u8; 5]; COUNT]) {
    let mut starts = [[NONE; 225]; 4];
    let mut cells = [[0; 5]; COUNT];
    let mut id = 0;
    let mut a = 0;
    while a < 4 {
        let (dx, dy) = AXES[a];
        let mut start = 0;
        while start < 225 {
            let (x, y) = ((start % 15) as i32, (start / 15) as i32);
            let (ex, ey) = (x + 4 * dx, y + 4 * dy);
            if ex >= 0 && ex < 15 && ey >= 0 && ey < 15 {
                let mut k = 0;
                while k < 5 {
                    cells[id][k] = ((y + k as i32 * dy) * 15 + x + k as i32 * dx) as u8;
                    k += 1;
                }
                starts[a][start] = id as u16;
                id += 1;
            }
            start += 1;
        }
        a += 1;
    }
    assert!(id == COUNT);
    (starts, cells)
}
const STARTS: ([[u16; 225]; 4], [[u8; 5]; COUNT]) = build_starts();
const CELLS: [[u8; 5]; COUNT] = STARTS.1;

// Windows through each point by axis, then by offset -4..=0, padded with NONE.
const THROUGH: [[u16; 20]; 225] = {
    let mut table = [[NONE; 20]; 225];
    let mut point = 0;
    while point < 225 {
        let mut n = 0;
        let mut a = 0;
        while a < 4 {
            let (dx, dy) = AXES[a];
            let mut offset = -4;
            while offset <= 0 {
                let x = (point % 15) as i32 + offset * dx;
                let y = (point / 15) as i32 + offset * dy;
                if x >= 0 && x < 15 && y >= 0 && y < 15 {
                    let id = STARTS.0[a][(y * 15 + x) as usize];
                    if id != NONE {
                        table[point][n] = id;
                        n += 1;
                    }
                }
                offset += 1;
            }
            a += 1;
        }
        point += 1;
    }
    table
};

fn through(point: usize) -> impl Iterator<Item = usize> {
    THROUGH[point]
        .iter()
        .take_while(|&&id| id != NONE)
        .map(|&id| usize::from(id))
}

#[derive(Default, Clone, Copy)]
pub struct Scan {
    pub value: u32,
    // Indexed by side: 0 the given player, 1 the opponent.
    pub five: [bool; 2],
    pub fours: [u8; 2],
}

pub struct Threat {
    pub value: u32,
    // The first legal winning reply, and whether a second distinct one exists.
    pub four: Option<usize>,
    pub fork: bool,
}

// Per window, the stones of each player; a blocked cell (3) counts for both.
pub struct Windows([[u8; 2]; COUNT]);
impl Windows {
    pub fn new(board: &Board) -> Self {
        let mut windows = Self([[0; 2]; COUNT]);
        for (i, &cell) in board.iter().enumerate() {
            match cell {
                1 | 2 => windows.place(i, cell),
                3 => {
                    windows.place(i, 1);
                    windows.place(i, 2);
                }
                _ => {}
            }
        }
        windows
    }
    pub fn place(&mut self, point: usize, player: u8) {
        for id in through(point) {
            self.0[id][usize::from(player - 1)] += 1;
        }
    }
    pub fn remove(&mut self, point: usize, player: u8) {
        for id in through(point) {
            self.0[id][usize::from(player - 1)] -= 1;
        }
    }
    fn open(&self, id: usize, player: u8) -> Option<u8> {
        let [own, enemy] = if player == 1 {
            self.0[id]
        } else {
            [self.0[id][1], self.0[id][0]]
        };
        (enemy == 0).then_some(own)
    }
    // One pass over the windows through an empty point for both players: the
    // combined score, open fives, and open fours (which may give winning replies).
    pub fn scan(&self, point: usize, player: u8) -> Scan {
        let mine = usize::from(player - 1);
        let mut scan = Scan::default();
        for id in through(point) {
            let counts = self.0[id];
            for (side, own, enemy) in [
                (0, counts[mine], counts[1 - mine]),
                (1, counts[1 - mine], counts[mine]),
            ] {
                if enemy != 0 {
                    continue;
                }
                scan.value += [0, 1, 12, 120, 10000][usize::from(own).min(4)];
                match own {
                    3 => scan.fours[side] += 1,
                    4.. => scan.five[side] = true,
                    _ => {}
                }
            }
        }
        scan
    }
    // Empty points of open windows holding `stones` of the player's stones,
    // each point once, in window order.
    fn gaps(&self, board: &Board, player: u8, stones: u8, out: &mut Vec<usize>) {
        for (id, cells) in CELLS.iter().enumerate() {
            if self.open(id, player) != Some(stones) {
                continue;
            }
            for &cell in cells {
                let cell = usize::from(cell);
                if board[cell] == 0 && !out.contains(&cell) {
                    out.push(cell);
                }
            }
        }
    }
    // Points where the player completes a legal five now.
    pub fn five_points(&self, board: &Board, player: u8, rule: u8) -> Vec<usize> {
        let mut points = Vec::new();
        self.gaps(board, player, 4, &mut points);
        points.retain(|&i| {
            rule == 0
                || (rule == 2 && player == 2)
                || (won(board, i, player, rule) && rules::legal(board, i, player, rule))
        });
        points
    }
    // Points that may make a four: an empty cell of an open window holding
    // three of the player's stones. `threat` confirms the winning reply.
    pub fn four_points(&self, board: &Board, player: u8) -> Vec<usize> {
        let mut points = Vec::new();
        self.gaps(board, player, 3, &mut points);
        points
    }
    // Points that may make a three: an empty cell of an open window holding
    // two of the player's stones.
    pub fn three_points(&self, board: &Board, player: u8) -> Vec<usize> {
        let mut points = Vec::new();
        self.gaps(board, player, 2, &mut points);
        points
    }
    // Cells of the open windows through `point` holding `stones` of the
    // player's stones.
    pub fn cells_through(&self, point: usize, player: u8, stones: u8) -> Vec<usize> {
        let mut cells = Vec::new();
        for id in through(point) {
            if self.open(id, player) == Some(stones) {
                for &cell in &CELLS[id] {
                    let cell = usize::from(cell);
                    if !cells.contains(&cell) {
                        cells.push(cell);
                    }
                }
            }
        }
        cells
    }
    // Whether a stone at the empty `point` makes five, checking exact-five rules on the board.
    pub fn wins(&self, board: &Board, point: usize, player: u8, rule: u8) -> bool {
        through(point).any(|id| self.open(id, player) == Some(4))
            && (rule == 0 || (rule == 2 && player == 2) || won(board, point, player, rule))
    }
    // Score every unblocked window advanced by a stone at the empty `point`, and
    // find the legal winning replies it creates.
    pub fn threat(&self, board: &Board, point: usize, player: u8, rule: u8) -> Threat {
        let mut result = Threat {
            value: 0,
            four: None,
            fork: false,
        };
        for id in through(point) {
            let Some(own) = self.open(id, player) else {
                continue;
            };
            let stones = usize::from(own) + 1;
            result.value += [0, 1, 12, 120, 10000][stones.min(5) - 1];
            if stones != 4 || result.fork {
                continue;
            }
            let gap = CELLS[id]
                .iter()
                .map(|&i| usize::from(i))
                .find(|&i| i != point && board[i] == 0)
                .unwrap();
            if result.four == Some(gap) {
                continue;
            }
            // In overline-allowed play the open window already proves the five.
            let legal_win = rule == 0 || (rule == 2 && player == 2) || {
                let mut after = *board;
                after[point] = player;
                won(&after, gap, player, rule) && rules::legal(&after, gap, player, rule)
            };
            if legal_win {
                if result.four.is_none() {
                    result.four = Some(gap);
                } else {
                    result.fork = true;
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_point_lies_in_its_windows() {
        for point in 0..225 {
            let ids: Vec<_> = through(point).collect();
            assert!((3..=20).contains(&ids.len()));
            assert!(ids.iter().all(|&id| CELLS[id].contains(&(point as u8))));
        }
        assert_eq!(through(112).count(), 20);
        assert_eq!(through(0).count(), 3);
    }
}
