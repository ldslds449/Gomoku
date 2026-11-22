use super::{Board, AXES};

// Cells five steps each way along every axis, with 255 off the board.
const NEIGHBORS: [[[u8; 11]; 4]; 225] = {
    let axes = [(1, 0), (0, 1), (1, 1), (1, -1)];
    let mut table = [[[255; 11]; 4]; 225];
    let mut point = 0;
    while point < 225 {
        let mut a = 0;
        while a < 4 {
            let mut k = 0;
            while k < 11 {
                let x = (point % 15) as i32 + axes[a].0 * (k as i32 - 5);
                let y = (point / 15) as i32 + axes[a].1 * (k as i32 - 5);
                if x >= 0 && x < 15 && y >= 0 && y < 15 {
                    table[point][a][k] = (y * 15 + x) as u8;
                }
                k += 1;
            }
            a += 1;
        }
        point += 1;
    }
    table
};

// Input already includes the new black stone.
pub fn forbidden(board: &Board, point: usize) -> bool {
    // Read five cells each way along every axis once; off-board cells read as 3.
    let lines = NEIGHBORS[point]
        .map(|cells| cells.map(|i| board.get(usize::from(i)).copied().unwrap_or(3)));
    // An overline needs five more black stones on one axis, and every double
    // needs at least two more on each of two axes.
    let nearby = lines.map(|line| line.iter().filter(|&&v| v == 1).count() - 1);
    if nearby.iter().all(|&n| n < 5) && nearby.iter().filter(|&&n| n >= 2).count() < 2 {
        return false;
    }
    let runs = lines.map(|line| {
        let left = line[..5].iter().rev().take_while(|&&v| v == 1).count();
        let right = line[6..].iter().take_while(|&&v| v == 1).count();
        left + 1 + right
    });
    if runs.iter().any(|&n| n > 5) {
        return true;
    }
    if runs.contains(&5) {
        return false;
    }
    let mut fours = 0;
    let mut threes = 0;
    for (cells, line) in NEIGHBORS[point].iter().zip(&lines) {
        let is = |step: i32, value| line[(step + 5) as usize] == value;
        let mut formations = [0u16; 5];
        let mut count = 0;
        for start in -4..=0 {
            let mut stones = 0u16;
            let mut gap = None;
            let mut valid = true;
            for step in start..start + 5 {
                match line[(step + 5) as usize] {
                    1 => stones |= 1 << (step + 4),
                    0 if gap.is_none() => gap = Some(step),
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }
            if valid && stones.count_ones() == 4 && !formations[..count].contains(&stones) {
                // `run` skips its own cell, so the gap counts as filled without a copy.
                let i = usize::from(cells[(gap.unwrap() + 5) as usize]);
                if AXES.iter().all(|&dir| super::run(board, i, 1, dir) <= 5) {
                    formations[count] = stones;
                    count += 1;
                }
            }
        }
        fours += count;
        // An open three fills its gap to a straight four: four consecutive stones
        // whose two open ends each complete exactly five, without an overline.
        for start in -3..=0 {
            let window = start..start + 4;
            if window.clone().filter(|&step| is(step, 1)).count() == 3
                && window.filter(|&step| is(step, 0)).count() == 1
                && is(start - 1, 0)
                && is(start + 4, 0)
                && !is(start - 2, 1)
                && !is(start + 5, 1)
            {
                threes += 1;
                break;
            }
        }
    }
    fours >= 2 || threes >= 2
}

pub fn legal(board: &Board, point: usize, player: u8, rule: u8) -> bool {
    if rule != 2 || player != 1 {
        return true;
    }
    let mut after = *board;
    after[point] = player;
    !forbidden(&after, point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_long_line_double_four_and_double_three() {
        for stones in [
            vec![110, 111, 113, 114, 115],
            vec![110, 111, 113, 82, 97, 127],
            vec![111, 113, 97, 127],
        ] {
            let mut board = [0; 225];
            for i in stones {
                board[i] = 1;
            }
            assert!(!legal(&board, 112, 1, 2));
            assert!(legal(&board, 112, 2, 2));
        }
        let mut board = [0; 225];
        for i in [110, 111, 113, 114] {
            board[i] = 1;
        }
        assert!(legal(&board, 112, 1, 2));
    }

    #[test]
    fn allows_four_three_and_ignores_threes_that_need_an_overline() {
        // Row 7 forms X_XXX (a four), column 7 forms an open three.
        let mut board = [0; 225];
        for i in [108, 110, 111, 97, 127] {
            board[i] = 1;
        }
        assert!(legal(&board, 112, 1, 2));
        // Row 7 B__B[B]B__B: either straight four would need an overline end.
        let mut board = [0; 225];
        for i in [108, 111, 113, 116, 97, 127] {
            board[i] = 1;
        }
        assert!(legal(&board, 112, 1, 2));
    }
}
