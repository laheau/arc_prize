"""Simple Sudoku dataset for training direction-based encoder-decoder models.

Provides clean puzzle-solution pairs as [9, 9] grids with values 0-9 (0 = blank).
"""

from __future__ import annotations

import csv

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset


class SudokuDataset(Dataset):
    """Simple Sudoku dataset from HuggingFace.
    
    Returns puzzle-solution pairs as [9, 9] tensors with values 0-9 (0 = blank).
    
    Args:
        split: 'train' or 'test'
        repo: HuggingFace dataset repo name
        min_difficulty: Optional minimum difficulty rating filter
    """
    
    def __init__(
        self,
        split: str = "train",
        repo: str = "sapientinc/sudoku-extreme",
        min_difficulty: int | None = None,
        limit: int | None = None,
    ):
        self.split = split
        self.puzzles = []
        self.solutions = []
        
        # Download and parse CSV from HuggingFace
        csv_path = hf_hub_download(repo, f"{split}.csv", repo_type="dataset")
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for i, (source, puzzle_str, solution_str, rating) in enumerate(reader):
                if limit is not None and len(self.puzzles) >= limit:
                    break
                
                if min_difficulty is None or int(rating) >= min_difficulty:
                    # Parse puzzle: '.' -> 0, '1'-'9' -> 1-9
                    puzzle = torch.tensor(
                        [int(c) if c != '.' else 0 for c in puzzle_str],
                        dtype=torch.long
                    ).view(9, 9)
                    
                    # Parse solution: '1'-'9' -> 1-9
                    solution = torch.tensor(
                        [int(c) for c in solution_str],
                        dtype=torch.long
                    ).view(9, 9)
                    
                    self.puzzles.append(puzzle)
                    self.solutions.append(solution)
    
    def __len__(self) -> int:
        return len(self.puzzles)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "puzzle": self.puzzles[idx],
            "solution": self.solutions[idx],
        }


def sudoku_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for Sudoku batches.
    
    Args:
        batch: List of dicts with 'puzzle' and 'solution' keys
        
    Returns:
        Dict with stacked tensors of shape [B, 9, 9]
    """
    return {
        "puzzle": torch.stack([item["puzzle"] for item in batch]),
        "solution": torch.stack([item["solution"] for item in batch]),
    }


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = SudokuDataset(split="train")
    print(f"Loaded {len(dataset)} puzzles")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=sudoku_collate_fn,
        num_workers=0,
    )
    
    # Test batch
    batch = next(iter(loader))
    print(f"Batch shapes: puzzle={batch['puzzle'].shape}, solution={batch['solution'].shape}")
    print(f"Puzzle value range: {batch['puzzle'].min()}-{batch['puzzle'].max()}")
    print(f"Solution value range: {batch['solution'].min()}-{batch['solution'].max()}")
    print("\nExample puzzle:")
    print(batch['puzzle'][0])
    print("\nExample solution:")
    print(batch['solution'][0])
