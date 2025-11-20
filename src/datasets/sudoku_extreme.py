from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.common import PuzzleDatasetMetadata


class SudokuExtremeDataset(Dataset):
    """Loads the processed Sudoku Extreme dataset saved in NumPy format."""

    def __init__(
        self,
        root: str | Path = "data/sudoku-extreme-full",
        split: str = "train",
        *,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        flatten: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split '{split}' not found under {self.root}. Run the preprocessing step first."
            )

        self.inputs = self._load_tensor(split_dir / "all__inputs.npy")
        self.labels = self._load_tensor(split_dir / "all__labels.npy")
        self.puzzle_indices = self._load_tensor(split_dir / "all__puzzle_indices.npy", dtype=torch.int32)
        self.group_indices = self._load_tensor(split_dir / "all__group_indices.npy", dtype=torch.int32)
        self.puzzle_identifiers = self._load_tensor(
            split_dir / "all__puzzle_identifiers.npy", dtype=torch.int32
        )

        self.metadata = PuzzleDatasetMetadata.load(split_dir / "dataset.json")
        self.transform = transform
        self.target_transform = target_transform
        self.flatten = flatten

    def _load_tensor(self, path: Path, dtype: torch.dtype | np.dtype | None = None) -> torch.Tensor:
        arr = np.load(path, mmap_mode=None)
        tensor = torch.from_numpy(arr.copy())
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        puzzle = self.inputs[idx] - 1  # original preprocessing stored values in 1..10
        solution = self.labels[idx] - 1

        if not self.flatten:
            puzzle = puzzle.view(9, 9)
            solution = solution.view(9, 9)
        mask = puzzle.ne(0)

        if self.transform is not None:
            puzzle = self.transform(puzzle)
        if self.target_transform is not None:
            solution = self.target_transform(solution)

        return {
            "puzzle": puzzle.long(),
            "solution": solution.long(),
            "given_mask": mask,
        }


__all__ = ["SudokuExtremeDataset"]
