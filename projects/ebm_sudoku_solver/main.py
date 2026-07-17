# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ebm-sudoku-solver",
# ]
# [tool.uv.sources]
# ebm-sudoku-solver = { path = "." }
# ///

from ebm_sudoku_solver.game.sudoku import (
    SudokuDifficulty,
    apply_mask,
    get_empty_board,
    get_is_valid_mask,
    get_random_mask,
    get_random_unverified_board,
    get_trivial_completed_board,
    is_valid_board,
    is_valid_solution,
    render,
)

from research.utils import seed_all


def main():
    print("EBM Sudoku Solver application script running!")
    seed_all()

    board_rank = 3
    empty_board = get_empty_board(board_rank=board_rank)
    is_valid_board(empty_board)
    print("Empty board:")
    print(render(empty_board))

    print("Trivial completed board:")
    trivial_board = get_trivial_completed_board(board_rank=board_rank)
    is_valid_board(trivial_board)
    is_valid_solution(trivial_board)
    print(render(trivial_board))

    print("Random unverified board:")
    random_board = get_random_unverified_board(board_rank=board_rank)
    is_valid_mask = get_is_valid_mask(random_board)
    print(render(random_board, mask=is_valid_mask))

    print("Easy trivial board to complete:")
    trivial_board = get_trivial_completed_board(board_rank=board_rank)
    mask = get_random_mask(difficulty=SudokuDifficulty.EASY, board_rank=board_rank)
    masked_board = apply_mask(trivial_board, mask)
    print(render(masked_board, mask=is_valid_mask))


if __name__ == "__main__":
    # uv run --script main.py
    # or when modifying the lib
    # uv run --script --reinstall main.py
    main()
