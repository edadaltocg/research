import pytest
import torch
from ebm_sudoku_solver.game.sudoku import render, verify_solution
from torch import Tensor


@pytest.fixture
def valid_board() -> Tensor:
    """Fixture that returns a valid 9x9 Sudoku board."""
    return torch.tensor([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ])


def test_verify_solution_valid(valid_board):
    """Test that a valid Sudoku board passes verification."""
    verify_solution(valid_board)


@pytest.mark.parametrize(
    "invalid_board_modifier, expected_error_msg",
    [
        # 1. Invalid shape
        (lambda board: torch.ones((5, 5)), "Board shape must be"),
        # 2. Duplicate elements in row (row sums to 45, but invalid)
        (
            lambda board: torch.cat(
                [
                    torch.tensor([[5, 5, 5, 5, 5, 5, 5, 5, 5]]),
                    board[1:],
                ],
                dim=0,
            ),
            None,
        ),
        # 3. Duplicate elements in column (swap elements)
        (
            lambda board: torch.tensor([
                [5, 3, 4, 6, 7, 8, 9, 1, 2],
                [6, 7, 2, 1, 9, 5, 3, 4, 8],
                [1, 9, 8, 3, 4, 2, 5, 6, 7],
                [8, 5, 9, 7, 6, 1, 4, 2, 3],
                [7, 2, 6, 8, 5, 3, 7, 9, 1],  # swapped 4 and 7 in the first column
                [4, 1, 3, 9, 2, 4, 8, 5, 6],
                [9, 6, 1, 5, 3, 7, 2, 8, 4],
                [2, 8, 7, 4, 1, 9, 6, 3, 5],
                [3, 4, 5, 2, 8, 6, 1, 7, 9],
            ]),
            None,
        ),
        # 4. Latin Square (invalid 3x3 blocks)
        (
            lambda board: torch.tensor([[(i + j) % 9 + 1 for j in range(9)] for i in range(9)]),
            None,
        ),
    ],
)
def test_verify_solution_invalid(valid_board, invalid_board_modifier, expected_error_msg):
    """Test that various invalid boards fail verification with an AssertionError."""
    invalid_board = invalid_board_modifier(valid_board)
    if expected_error_msg:
        with pytest.raises(AssertionError, match=expected_error_msg):
            verify_solution(invalid_board)
    else:
        with pytest.raises(AssertionError):
            verify_solution(invalid_board)


def test_render(valid_board):
    """Test that render function prints the board in correct ASCII format."""
    rendered = render(valid_board)
    lines = rendered.splitlines()
    assert lines[0] == "+-------+-------+-------+"
    assert lines[1] == "| 5 3 4 | 6 7 8 | 9 1 2 |"
    assert lines[2] == "| 6 7 2 | 1 9 5 | 3 4 8 |"
    assert lines[3] == "| 1 9 8 | 3 4 2 | 5 6 7 |"
    assert lines[4] == "+-------+-------+-------+"
