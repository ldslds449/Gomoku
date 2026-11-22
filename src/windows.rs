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

// Per-point totals of the windows through each point, as `Windows::scan`
// reports them, indexed by player (0 black, 1 white).
pub struct PointStats {
    value: [u32; 225],
    five: [[bool; 2]; 225],
    fours: [[u8; 2]; 225],
}
impl PointStats {
    pub fn scan(&self, point: usize, player: u8) -> Scan {
        let (mine, theirs) = if player == 1 { (0, 1) } else { (1, 0) };
        Scan {
            value: self.value[point],
            five: [self.five[point][mine], self.five[point][theirs]],
            fours: [self.fours[point][mine], self.fours[point][theirs]],
        }
    }
}

const WORDS: usize = COUNT.div_ceil(64);
const WEIGHTS: [u32; 5] = [0, 1, 12, 120, 10000];

// Per window, the stones of each player; a blocked cell (3) counts for both.
// Bit sets mirror the counts so that scans visit only the windows that matter:
// `open[p][k - 2]` holds the windows open for player p with k of p's stones
// (k = 2..=4), and `live` the windows holding stones of only one player.
#[derive(Clone)]
pub struct Windows {
    counts: [[u8; 2]; COUNT],
    open: [[[u64; WORDS]; 3]; 2],
    live: [u64; WORDS],
}
impl Windows {
    pub fn new(board: &Board) -> Self {
        let mut windows = Self {
            counts: [[0; 2]; COUNT],
            open: [[[0; WORDS]; 3]; 2],
            live: [0; WORDS],
        };
        // Count first, then file each window once.
        for (i, &cell) in board.iter().enumerate() {
            if matches!(cell, 1..=3) {
                for id in through(i) {
                    if cell != 2 {
                        windows.counts[id][0] += 1;
                    }
                    if cell != 1 {
                        windows.counts[id][1] += 1;
                    }
                }
            }
        }
        for id in 0..COUNT {
            if windows.counts[id] != [0, 0] {
                windows.toggle(id);
            }
        }
        windows
    }
    // Flip the bits that the window's current counts set; called before and
    // after a count changes, it moves the window to its new sets.
    fn toggle(&mut self, id: usize) {
        let (word, bit) = (id / 64, 1u64 << (id % 64));
        let [black, white] = self.counts[id];
        for (side, own, enemy) in [(0, black, white), (1, white, black)] {
            if enemy == 0 && (2..=4).contains(&own) {
                self.open[side][usize::from(own - 2)][word] ^= bit;
            }
        }
        if (black == 0) != (white == 0) {
            self.live[word] ^= bit;
        }
    }
    pub fn place(&mut self, point: usize, player: u8) {
        for id in through(point) {
            self.toggle(id);
            self.counts[id][usize::from(player - 1)] += 1;
            self.toggle(id);
        }
    }
    pub fn remove(&mut self, point: usize, player: u8) {
        for id in through(point) {
            self.toggle(id);
            self.counts[id][usize::from(player - 1)] -= 1;
            self.toggle(id);
        }
    }
    fn open(&self, id: usize, player: u8) -> Option<u8> {
        let [own, enemy] = if player == 1 {
            self.counts[id]
        } else {
            [self.counts[id][1], self.counts[id][0]]
        };
        (enemy == 0).then_some(own)
    }
    // The windows of a bit set in increasing order.
    fn ids(set: &[u64; WORDS]) -> impl Iterator<Item = usize> + '_ {
        set.iter().enumerate().flat_map(|(word, &bits)| {
            let mut bits = bits;
            std::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    word * 64 + bit
                })
            })
        })
    }
    // `scan` for every point at once: each window holding stones of one player
    // adds its weight to its five cells.
    pub fn point_stats(&self) -> PointStats {
        let mut stats = PointStats {
            value: [0; 225],
            five: [[false; 2]; 225],
            fours: [[0; 2]; 225],
        };
        for id in Self::ids(&self.live) {
            let [black, white] = self.counts[id];
            let (side, own) = if white == 0 { (0, black) } else { (1, white) };
            let weight = WEIGHTS[usize::from(own).min(4)];
            for &cell in &CELLS[id] {
                let cell = usize::from(cell);
                stats.value[cell] += weight;
                match own {
                    3 => stats.fours[cell][side] += 1,
                    4.. => stats.five[cell][side] = true,
                    _ => {}
                }
            }
        }
        stats
    }
    // One pass over the windows through an empty point for both players: the
    // combined score, open fives, and open fours (which may give winning
    // replies). `point_stats` gives the same for every point at once.
    #[cfg(test)]
    pub fn scan(&self, point: usize, player: u8) -> Scan {
        let mine = usize::from(player - 1);
        let mut scan = Scan::default();
        for id in through(point) {
            let counts = self.counts[id];
            for (side, own, enemy) in [
                (0, counts[mine], counts[1 - mine]),
                (1, counts[1 - mine], counts[mine]),
            ] {
                if enemy != 0 {
                    continue;
                }
                scan.value += WEIGHTS[usize::from(own).min(4)];
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
        let set = &self.open[usize::from(player - 1)][usize::from(stones - 2)];
        for id in Self::ids(set) {
            for &cell in &CELLS[id] {
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
    // Windows through `point` as (id, the player's stones), skipping those
    // holding the opponent's.
    pub fn open_through(&self, point: usize, player: u8) -> impl Iterator<Item = (usize, u8)> + '_ {
        through(point).filter_map(move |id| self.open(id, player).map(|own| (id, own)))
    }
    pub fn cells(id: usize) -> impl Iterator<Item = usize> {
        CELLS[id].iter().map(|&cell| usize::from(cell))
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
            result.value += WEIGHTS[stones.min(5) - 1];
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
    // After random placements and removals, the bit sets and the per-point
    // totals agree with counts rebuilt from the board.
    #[test]
    fn bit_sets_and_point_totals_follow_the_counts() {
        let mut rng = 7u32;
        let mut next = |n: u32| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng % n
        };
        let mut board = [0; 225];
        let mut windows = Windows::new(&board);
        for step in 0..3000 {
            let i = next(225) as usize;
            match board[i] {
                0 => {
                    let p = 1 + next(2) as u8;
                    board[i] = p;
                    windows.place(i, p);
                }
                p => {
                    board[i] = 0;
                    windows.remove(i, p);
                }
            }
            if step % 50 != 0 {
                continue;
            }
            let fresh = Windows::new(&board);
            assert_eq!(windows.counts, fresh.counts);
            assert_eq!(windows.open, fresh.open);
            assert_eq!(windows.live, fresh.live);
            let stats = windows.point_stats();
            for point in (0..225).filter(|&i| board[i] == 0) {
                for p in [1, 2] {
                    let (a, b) = (stats.scan(point, p), windows.scan(point, p));
                    assert_eq!((a.value, a.five, a.fours), (b.value, b.five, b.fours));
                }
            }
            for p in [1, 2] {
                for stones in 2..=4 {
                    let mut expected = Vec::new();
                    for (id, cells) in CELLS.iter().enumerate() {
                        if windows.open(id, p) == Some(stones) {
                            for &cell in cells {
                                let cell = usize::from(cell);
                                if board[cell] == 0 && !expected.contains(&cell) {
                                    expected.push(cell);
                                }
                            }
                        }
                    }
                    let mut found = Vec::new();
                    windows.gaps(&board, p, stones, &mut found);
                    assert_eq!(found, expected);
                }
            }
        }
    }
}
