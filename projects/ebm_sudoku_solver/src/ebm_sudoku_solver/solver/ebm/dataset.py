"""
Dataset for traninig ML models to solve the sudoku puzzle.

Self-supervised task.

Start from a complete board and mask elements.

Supervised dataset of (X, Y) pairs, conditioned by difficulty.
"""

import torch
from torch import Tensor

from ebm_sudoku_solver.game.sudoku import DEFAULT_BOARD_RANK, DEFAULT_EMPTY_CELL_VALUE, SudokuDifficulty, suggest_board


class SudokuDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        length: int = 1_000,
        difficulty: SudokuDifficulty = SudokuDifficulty.EASY,
        *,
        board_rank: int = DEFAULT_BOARD_RANK,
        empty_value: int = DEFAULT_EMPTY_CELL_VALUE,
        generator: torch.Generator | None = None,
    ) -> None:
        self.difficulty = difficulty
        self.length = length
        self.board_rank = board_rank
        self.empty_value = empty_value

        if not generator:
            self.generator = torch.Generator()
        else:
            self.generator = generator
        super().__init__()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        x, y = suggest_board(
            self.difficulty, generator=self.generator, empty_value=self.empty_value, board_rank=self.board_rank
        )
        return {"x": x, "y": y, "difficulty": torch.tensor([self.difficulty.value], dtype=torch.long)}


if __name__ == "__main__":
    dataset = SudokuDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    print(f"{dataset=}")
    print(f"{len(dataset)=}")
    for batch in dataloader:
        continue
