# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ebm-sudoku-solver",
# ]
# [tool.uv.sources]
# ebm-sudoku-solver = { path = "." }
# ///

import torch
from ebm_sudoku_solver.game.sudoku import (
    SudokuDifficulty,
    _rotate_board_clockwise_90_deg,
    apply_mask,
    count_empty_cells,
    get_empty_board,
    get_is_valid_mask,
    get_random_mask,
    get_random_unverified_board,
    get_trivial_completed_board,
    is_valid_board,
    is_valid_solution,
    render,
    solve,
    suggest_board,
    verify_solution,
    _permutate_rows_within_blocks,
    _permutate_cols_within_blocks,
)

from research.utils import seed_all


def main():
    board_rank = 3
    difficulty = SudokuDifficulty.EASY

    print("EBM Sudoku Solver application script running!")
    generator = torch.Generator()
    seed_all()

    empty_board = get_empty_board(board_rank=board_rank)
    is_valid_board(empty_board)
    print("Empty board:")
    print(render(empty_board))
    print(f"Empty cells: {count_empty_cells(empty_board)}")

    print("Trivial completed board:")
    trivial_board = get_trivial_completed_board(board_rank=board_rank)
    is_valid_board(trivial_board)
    is_valid_solution(trivial_board)
    print(render(trivial_board))

    print("Random unverified board:")
    random_board = get_random_unverified_board(board_rank=board_rank)
    is_valid_mask = get_is_valid_mask(random_board)
    print(render(random_board, mask=is_valid_mask))

    print("Trivial board to complete:")
    trivial_board = get_trivial_completed_board(board_rank=board_rank)
    mask = get_random_mask(difficulty=difficulty, board_rank=board_rank)
    masked_board = apply_mask(trivial_board, mask)
    is_valid_mask = get_is_valid_mask(masked_board)
    print(render(masked_board, mask=is_valid_mask))
    print(f"Empty cells: {count_empty_cells(masked_board)}")

    print("Rotated trivial board")
    trivial_board = get_trivial_completed_board(board_rank=board_rank)
    rotated_board = _rotate_board_clockwise_90_deg(trivial_board)
    print(render(rotated_board))

    print("Blocks symmetry")
    blocks_tensor = trivial_board.reshape(board_rank, board_rank, board_rank, board_rank)
    print(render(trivial_board))

    print("Get incomplete (valid) board")
    incomplete_board, _ = suggest_board(difficulty=difficulty, board_rank=board_rank, generator=generator)
    is_valid_mask = get_is_valid_mask(incomplete_board)
    print(render(incomplete_board, mask=is_valid_mask))

    print("Solve board")
    print("\tStarting board")
    print(render(incomplete_board, mask=is_valid_mask))
    print("\tSolved board")
    solved_board = solve(incomplete_board)
    is_valid_mask = get_is_valid_mask(solved_board)
    is_valid_solution(solved_board)
    print(render(solved_board, mask=is_valid_mask))
    assert verify_solution(incomplete_board, solved_board)

    print("Permutation tests")
    new_board = _permutate_rows_within_blocks(trivial_board)
    is_valid_mask = get_is_valid_mask(trivial_board)
    print(render(trivial_board, mask=is_valid_mask))
    is_valid_mask = get_is_valid_mask(new_board)
    print(render(new_board, mask=is_valid_mask))

    new_board = _permutate_cols_within_blocks(trivial_board)
    is_valid_mask = get_is_valid_mask(trivial_board)
    print(render(trivial_board, mask=is_valid_mask))
    is_valid_mask = get_is_valid_mask(new_board)
    print(render(new_board, mask=is_valid_mask))


if __name__ == "__main__":
    # uv run --script main.py
    # or when modifying the lib
    # uv run --script --reinstall main.py
    main()
