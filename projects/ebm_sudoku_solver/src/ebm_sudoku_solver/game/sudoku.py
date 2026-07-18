"""
Module to define analytical utilities for the Sudoku game.

"""

import math
from enum import IntEnum
from functools import partial, reduce

import torch
from loguru import logger
from torch import Generator, Tensor

# TODO: make them env vars
DEFAULT_BOARD_RANK = 3
DEFAULT_EMPTY_CELL_VALUE = 0
STRIKE_THROUGH_CHAR = "\u0336"
DEFAULT_EMPTY_CELL_CHAR = "."


class SudokuDifficulty(IntEnum):
    EASY = 41
    MEDIUM = 30
    HARD = 22
    MASTER = 17  # TODO: Refer to proof paper.


def _fmt_cell(
    cell: int, cell_w: int, width: int, *, is_valid: bool = True, empty_cell_char: str = DEFAULT_EMPTY_CELL_CHAR
) -> str:
    if 1 <= cell <= width and is_valid:
        return str(cell).rjust(cell_w)
    if 1 <= cell <= width and not is_valid:
        return str(cell).rjust(cell_w) + "\u0336"
    else:
        return empty_cell_char.rjust(cell_w)


def render(board: Tensor, *, mask: Tensor | None = None) -> str:
    """
    Visualization utility.

    Renders Sudoku board as ASCII.
    """
    mask_ = torch.ones_like(board, dtype=torch.bool)
    if mask is not None:
        mask_ = mask
    board_ = board.tolist()
    width = get_board_width(board)
    rank = get_board_rank(board)
    cell_w = len(str(width))  # width per cell for alignment

    # A separator sized to the board: e.g. "+-------+-------+-------+"
    block_dashes = "-" * (rank * (cell_w + 1) + 1)
    sep = "+" + "+".join([block_dashes] * rank) + "+"

    lines = []
    for i, row in enumerate(board_):
        if i % rank == 0:
            lines.append(sep)
        cells = [_fmt_cell(c, cell_w, width, is_valid=bool(mask_[i][j].item())) for j, c in enumerate(row)]
        blocks = [" ".join(cells[j : j + rank]) for j in range(0, width, rank)]
        lines.append("| " + " | ".join(blocks) + " |")
    lines.append(sep)
    return "\n".join(lines)


def to_board_type(board: list[list[int]]) -> Tensor:
    return torch.tensor(board, dtype=torch.long)


def get_empty_board(board_rank: int = DEFAULT_BOARD_RANK) -> Tensor:
    return torch.zeros(size=(board_rank**2, board_rank**2), dtype=torch.long)


def get_random_unverified_board(board_rank: int = DEFAULT_BOARD_RANK) -> Tensor:
    return torch.randint(low=1, high=board_rank**2 + 1, size=(board_rank**2, board_rank**2), dtype=torch.long)


def get_board_width(board: Tensor) -> int:
    return len(board)


def get_board_rank(board: Tensor) -> int:
    return int(math.sqrt(get_board_width(board)))


def count_empty_cells(board: Tensor, *, empty_value: int = DEFAULT_EMPTY_CELL_VALUE) -> int:
    return int(torch.sum(board == empty_value).to(dtype=torch.long).item())


def get_trivial_completed_board(board_rank: int = DEFAULT_BOARD_RANK) -> Tensor:
    initial_board = to_board_type(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 1, 5, 6, 4, 8, 9, 7],
            [5, 6, 4, 8, 9, 7, 2, 3, 1],
            [8, 9, 7, 2, 3, 1, 5, 6, 4],
            [3, 1, 2, 6, 4, 5, 9, 7, 8],
            [6, 4, 5, 9, 7, 8, 3, 1, 2],
            [9, 7, 8, 3, 1, 2, 6, 4, 5],
        ],
    )
    return initial_board


def is_valid_solution(board: Tensor):
    """
    Verify complete sudoku board.

    For a board of rank 3:

    Every number between 1 to 9 appears only once in every row, column and 3X3 sub-matrix of the sudoku.

    Args:
        board (Tensor): N x N tensor with long dtype.
    """

    width = get_board_width(board)
    rank = get_board_rank(board)

    assert board.shape == (width, width), f"Board shape must be ({width}, {width}), but got {board.shape}"

    expected = torch.arange(1, width + 1, dtype=board.dtype, device=board.device)

    # 1. Verify rows
    rows_sorted, _ = torch.sort(board, dim=1)
    torch.testing.assert_close(rows_sorted, expected.expand(width, width))

    # 2. Verify columns
    cols_sorted, _ = torch.sort(board.T, dim=1)
    torch.testing.assert_close(cols_sorted, expected.expand(width, width))

    # 3. Verify blocks
    # stretch them into lines to check similarly to the columns and rows
    # (block per line, row within block, block per column, column within block) -- before permutation
    # (block per line, block per column, row within block, column within block) -- after permutation
    blocks = board.reshape(rank, rank, rank, rank).permute(0, 2, 1, 3).reshape(width, width)
    blocks_sorted, _ = torch.sort(blocks, dim=1)
    torch.testing.assert_close(blocks_sorted, expected.expand(width, width))


def is_valid_board(board: Tensor, *, empty_value: int = DEFAULT_EMPTY_CELL_VALUE) -> bool:
    """
    Verify validity of any incomplete sudoku board.

    Constraints:
        For a board with rank 3:
        1. Each row must contain the digits 1-9 without repetition.
        2. Each column must contain the digits 1-9 without repetition.
        3. Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

    Algorithm:

    Complexity:
        O(N^2)
    """
    width = get_board_width(board)
    rank = get_board_rank(board)
    board_: list[list[int]] = board.tolist()
    for i in range(width):
        values_in_row = set()
        values_in_col = set()
        values_in_blk = set()
        for j in range(width):
            logger.debug(f"row {(i, j)}")
            logger.debug(f"col {(j, i)}")
            k_row = rank * (i // rank) + j // rank
            k_col = rank * (i % rank) + j % rank
            logger.debug(f"blk {(k_row, k_col)}")

            val_row = board_[i][j]
            val_col = board_[j][i]
            val_blk = board_[k_row][k_col]

            # 1. Check rows
            if val_row in values_in_row and val_row != empty_value:
                return False
            # 2. Check cols
            if val_col in values_in_col and val_col != empty_value:
                return False
            # 3. Check blocks
            if val_blk in values_in_blk and val_blk != empty_value:
                return False
            values_in_row.add(val_row)
            values_in_col.add(val_col)
            values_in_blk.add(val_blk)

    return True


def get_is_valid_mask(board: Tensor, *, empty_value: int = DEFAULT_EMPTY_CELL_VALUE) -> Tensor:
    width = get_board_width(board)
    rank = get_board_rank(board)
    board_: list[list[int]] = board.tolist()

    mask = torch.ones_like(board, dtype=torch.bool)
    for i in range(width):
        values_in_row = set()
        values_in_col = set()
        values_in_blk = set()
        for j in range(width):
            logger.debug(f"row {(i, j)}")
            logger.debug(f"col {(j, i)}")
            k_row = rank * (i // rank) + j // rank
            k_col = rank * (i % rank) + j % rank
            logger.debug(f"blk {(k_row, k_col)}")

            val_row = board_[i][j]
            val_col = board_[j][i]
            val_blk = board_[k_row][k_col]

            # 1. Check rows
            if val_row in values_in_row and val_row != empty_value:
                mask[i][j] = False
            # 2. Check cols
            if val_col in values_in_col and val_col != empty_value:
                mask[j][i] = False
            # 3. Check blocks
            if val_blk in values_in_blk and val_blk != empty_value:
                mask[k_row][k_col] = False
            values_in_row.add(val_row)
            values_in_col.add(val_col)
            values_in_blk.add(val_blk)

    return mask


def solve(board: Tensor, *, empty_value: int = DEFAULT_EMPTY_CELL_VALUE) -> Tensor:
    """
    Solve a Sudoku board with a backtracking algorithm.

    Note that the board may contain more than one solution.
    This algorithm will find one of them, not all of them.

    PS.:
        Time sensitive, might benefit from a Rust implementation.
    """
    width = get_board_width(board)
    rank = get_board_rank(board)

    board_: list[list[int]] = board.tolist()

    # constraints
    rows_c = [set() for _ in range(width)]
    cols_c = [set() for _ in range(width)]
    blks_c = [set() for _ in range(width)]

    empty_indices = []

    for i in range(width):
        for j in range(width):
            v = board_[i][j]
            if v == empty_value:
                empty_indices.append((i, j))
            else:
                k = i // rank * rank + j // rank
                rows_c[i].add(v)
                cols_c[j].add(v)
                blks_c[k].add(v)

    digits = set(range(1, width + 1))

    def _solve(idx: int) -> bool:
        if idx == len(empty_indices):
            return True
        i, j = empty_indices[idx]
        k = i // rank * rank + j // rank
        for v in digits - rows_c[i] - cols_c[j] - blks_c[k]:
            board_[i][j] = v
            rows_c[i].add(v)
            cols_c[j].add(v)
            blks_c[k].add(v)

            if _solve(idx + 1):
                return True

            # backtracking cleanup
            board_[i][j] = empty_value
            rows_c[i].discard(v)
            cols_c[j].discard(v)
            blks_c[k].discard(v)

        return False

    solved = _solve(0)
    if not solved:
        raise ValueError("Invalid board!")

    return to_board_type(board_)


def _rotate_board_clockwise_90_deg(board: Tensor, *, batched: bool = False) -> Tensor:
    dims = (0, 1) if not batched else (1, 2)
    return board.rot90(k=-1, dims=dims)


def _rotate_board_clockwise_180_deg(board: Tensor, *, batched: bool = False) -> Tensor:
    dims = (0, 1) if not batched else (1, 2)
    return board.rot90(k=-2, dims=dims)


def _rotate_board_clockwise_270_deg(board: Tensor, *, batched: bool = False) -> Tensor:
    dims = (0, 1) if not batched else (1, 2)
    return board.rot90(k=-3, dims=dims)


def _reflect_board_horizontally(board: Tensor, *, batched: bool = False) -> Tensor:
    # torch.flip creates a copy, better is np flip
    dims = (0, 1) if not batched else (1, 2)
    # force=False is a memory view instead of copy
    return torch.flip(board, dims=dims)


def _reflect_board_vertically(board: Tensor, *, batched: bool = False) -> Tensor:
    return torch.fliplr(board)


def _reflect_board_diagonally(board: Tensor, *, batched: bool = False) -> Tensor:
    """Reflect across the main diagonal:  (i, j) -> (j, i)."""
    return board.T


def _reflect_board_antidiagonally(board: Tensor, *, batched: bool = False) -> Tensor:
    """Reflect across the anti-diagonal:  (i, j) -> (n-1-j, n-1-i)."""
    return board.flip((0, 1)).T


def _relabel_digits(
    board: Tensor, *, empty_value: int = DEFAULT_EMPTY_CELL_VALUE, generator: Generator | None = None
) -> Tensor:
    """Apply a random bijection of the digits {1, ..., width}.

    Number of possible combinations: 9!
    """
    width = get_board_width(board)
    perm = torch.randperm(width, device=board.device, generator=generator) + 1
    lookup = torch.cat([
        torch.ones(1, dtype=board.dtype, device=board.device) * empty_value,
        perm.to(board.dtype),
    ])
    # broadcasting to index shape. Same bijection on entire batch.
    return lookup[board]


def _permutate_rows_within_blocks(
    board: Tensor, *, batched: bool = False, generator: Generator | None = None
) -> Tensor:
    """Number of possible permutations: 9!"""
    width = get_board_width(board)
    perm = torch.randperm(width, generator=generator)
    if batched:
        return board[:, perm]
    return board[perm]


def _permutate_cols_within_blocks(
    board: Tensor, *, batched: bool = False, generator: Generator | None = None
) -> Tensor:
    """Number of possible permutations: 9!"""
    width = get_board_width(board)
    perm = torch.randperm(width, generator=generator)
    if batched:
        return board[:, :, perm]
    return board[:, perm]


def _permutate_blocks_rows(board: Tensor, *, batched: bool = False, generator: Generator | None = None) -> Tensor:
    """Number of possible permutations: 3!
    (0,1,2)
    (1,0,2)
    (1,2,0)
    (0,2,1)
    (2,0,1)
    (2,1,0)
    """
    rank = get_board_rank(board)
    width = get_board_width(board)
    perm = torch.randperm(rank, generator=generator)
    # (block per line, row within block, block per column, column within block)
    board = board.reshape(rank, rank, rank, rank)
    if batched:
        return board[:, perm].reshape(width, width)
    return board[perm].reshape(width, width)


def _permutate_blocks_cols(board: Tensor, *, batched: bool = False, generator: Generator | None = None) -> Tensor:
    """Number of possible permutations: 3!"""
    rank = get_board_rank(board)
    width = get_board_width(board)
    perm = torch.randperm(rank, generator=generator)
    # (block per line, row within block, block per column, column within block)
    board = board.reshape(rank, rank, rank, rank)
    if batched:
        return board[:, :, :, perm].reshape(width, width)
    return board[:, :, perm].reshape(width, width)


def augment(
    board: Tensor,
    *,
    empty_value: int = DEFAULT_EMPTY_CELL_VALUE,
    batched: bool = False,
    generator: Generator | None = None,
) -> Tensor:
    rotations = {
        0: _rotate_board_clockwise_90_deg,
        1: _rotate_board_clockwise_180_deg,
        2: _rotate_board_clockwise_270_deg,
    }
    _rotate = rotations[int(torch.randint(0, len(rotations), (1,), generator=generator).item())]

    flips = {
        0: _reflect_board_horizontally,
        1: _reflect_board_vertically,
        2: _reflect_board_diagonally,
        3: _reflect_board_antidiagonally,
    }
    _flip = flips[int(torch.randint(0, len(flips), (1,), generator=generator).item())]

    # 3*4*9!*3!*3!*3!*3! = 5 643 509 760 = 5.6 * 10^9
    transforms = [
        partial(_rotate, batched=batched),  # 3
        partial(_flip, batched=batched),  # 4
        partial(_relabel_digits, empty_value=empty_value),  # 9!
        partial(_permutate_blocks_rows, batched=batched, generator=generator),  # 3!
        partial(_permutate_blocks_cols, batched=batched, generator=generator),  # 3!
        # TODO:
        # partial(_permutate_rows_within_blocks, batched=batched, generator=generator),  # 3!
        # partial(_permutate_cols_within_blocks, batched=batched, generator=generator),  # 3!
    ]

    board = reduce(lambda b, func: func(b), transforms, board)
    return board


def get_random_mask(
    difficulty: SudokuDifficulty = SudokuDifficulty.EASY, board_rank: int = DEFAULT_BOARD_RANK
) -> Tensor:
    """O(n)."""
    # TODO: speed this up, batch it up
    width = board_rank**2
    mask_size = width**2 - difficulty.value
    # unique random indexes in the board
    key_mask = torch.randperm(width**2, dtype=torch.long)[:mask_size]
    logger.debug(f"{key_mask=}")
    logger.debug(f"{key_mask.shape=}")
    mask = torch.zeros((width, width), dtype=torch.bool)
    logger.debug(f"{mask=}")
    for k in key_mask:
        logger.debug(f"{k=}")
        i = k // width
        logger.debug(f"{i=}")
        j = k % width
        logger.debug(f"{j=}")
        mask[i][j] = True
    return mask


def apply_mask(board: Tensor, mask: Tensor, *, empty_value: int = DEFAULT_EMPTY_CELL_VALUE) -> Tensor:
    return board.masked_fill(mask, empty_value)


def has_unique_solution(board: Tensor, *, empty_value: int = DEFAULT_EMPTY_CELL_VALUE) -> bool:
    # TODO: improve this! Do I need to fully solve to find unique solution?
    try:
        board = solve(board, empty_value=empty_value)
    except ValueError as e:
        logger.error(e)
        return False

    return True


def suggest_board(
    difficulty: SudokuDifficulty = SudokuDifficulty.EASY,
    *,
    board_rank: int = DEFAULT_BOARD_RANK,
    empty_value: int = DEFAULT_EMPTY_CELL_VALUE,
    generator: Generator | None = None,
) -> Tensor:
    """
    Generation is much harder than verification.

    Number of unique sudoku boards:
        N = 6670903752021072936960
        or 6.671*10^21
        with a lot of symmetries, which reduces this numbers to essentially
        N = 5472730538 or 5.473*10^9.
        different Sudoku games.

    Symmetries:
        1. Relabeling the nine digits.
        2. Permuting the three stacks.
        3. Permuting the three bands.
        4. Permuting the three columns within a stack.
        5. Permuting the three rows within a band.
        6. Any reflection or rotation (from the list of symmetries of a square below).

    Symmetries of a square:
        1. Rotation by 0 degrees (the identity transformation).
        2. Rotation clockwise by 90 degrees.
        3. Rotation clockwise by 180 degrees.
        4. Rotation clockwise by 270 degrees.
        5. Reflection in the horizontal axis (through the center of the square).
        6. Reflection in the vertical axis (through the center of the square).
        7. Reflection in the diagonal from the bottom left to the upper right corner.
        8. Reflection in the diagonal from the upper left to the bottom right corner.

    Constraints:
        1. You cannot build a Sudoku with more than nine empty 9-number groups (rows, columns, or 3x3 blocks)
        TODO: Is this to have unique solution?

    Algorithm:
        1. Seed a complete board.
        for i=1..(board_size ** 4 - difficulty):
            2. Suggest one element to mask.
            3. Check for solution with backtracking solver.
            4. If not unique solution: Go to 2.
            5. If unique solution: Continue

    PS.:
        Time sensitive, might benefit from a Rust implementation.
    """
    board = get_trivial_completed_board(board_rank=board_rank)
    board = augment(board, empty_value=empty_value, generator=generator)
    mask = get_random_mask(difficulty=difficulty, board_rank=board_rank)
    masked_board = apply_mask(board, mask, empty_value=empty_value)

    _counter = 1
    while not has_unique_solution(masked_board):
        board = get_trivial_completed_board(board_rank=board_rank)
        mask = get_random_mask(difficulty=difficulty, board_rank=board_rank)
        masked_board = apply_mask(board, mask, empty_value=empty_value)
        _counter += 1

    logger.debug(f"Tried {_counter} board(s) before finding a valid one.")
    return masked_board


def verify_solution(board: Tensor, solved_board: Tensor) -> bool:
    """
    Verify if solved board is the solution of board.

    To be a solution:
        1. every filed in cell in board should be a match in solved board;
        2. Solved board should be a valid solution;
    """
    # TODO
    return False


def information_content():
    # TODO: can we store information in a sudoku game?
    # Sudoku compression?
    # How to store all boards in memory? Tree of depth 81?
    return
