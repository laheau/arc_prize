"""
Paper Reproduction Package
==========================

This package implements key techniques from the research paper for ARC tasks.

Main components:
- DeepRecursiveModel: Core recursive model with gradient detachment
- ARCDeepRecursiveModel: Wrapper for ARC tasks
- train: Memory-efficient training utilities
"""

from .model import (
    DeepRecursiveModel,
    ARCDeepRecursiveModel,
    count_parameters
)

from .train import (
    train,
    train_epoch,
    train_step_memory_efficient,
    train_step_standard,
    evaluate,
    EMA
)

__all__ = [
    'DeepRecursiveModel',
    'ARCDeepRecursiveModel',
    'count_parameters',
    'train',
    'train_epoch',
    'train_step_memory_efficient',
    'train_step_standard',
    'evaluate',
    'EMA',
]

__version__ = '0.1.0'
