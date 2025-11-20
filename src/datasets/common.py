from __future__ import annotations

from pathlib import Path
from typing import List, Union

from pydantic import BaseModel


PathLike = Union[str, Path]


class PuzzleDatasetMetadata(BaseModel):
	"""Lightweight description for tokenized puzzle datasets."""

	seq_len: int
	vocab_size: int
	pad_id: int
	ignore_label_id: int
	blank_identifier_id: int
	num_puzzle_identifiers: int
	total_groups: int
	mean_puzzle_examples: int
	total_puzzles: int
	sets: List[str]

	@classmethod
	def load(cls, path: PathLike) -> "PuzzleDatasetMetadata":
		with open(path, "r", encoding="utf-8") as f:
			return cls.model_validate_json(f.read())

	def save(self, path: PathLike) -> None:
		with open(path, "w", encoding="utf-8") as f:
			f.write(self.model_dump_json(indent=2))
