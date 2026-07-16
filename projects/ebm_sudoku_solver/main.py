# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ebm-sudoku-solver",
# ]
# [tool.uv.sources]
# ebm-sudoku-solver = { path = "." }
# ///

from ebm_sudoku_solver.game.sudoku import (
    get_empty_board,
    get_trivial_completed_board,
    is_valid_board,
    render,
    is_valid_solution,
)


def main():
    print("EBM Sudoku Solver application script running!")

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


if __name__ == "__main__":
    # uv run --script main.py
    # or when modifying the lib
    # uv run --script --reinstall main.py
    main()
