from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, Tuple, List
import numpy as np
import json


DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr
    
    
def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


def color_permutation(arr: np.ndarray, perm) -> np.ndarray:
    """Apply color permutation to the input array."""
    permuted_arr = np.copy(arr)
    for original_color, new_color in enumerate(perm):
        permuted_arr[arr == original_color] = new_color
    return permuted_arr


def pad_frames(input_frame, output_frame, target_size=32, pad_value=0, indicator_value=1):
    """Pad input and output frames to target size with random offset."""
    input_frame_indicator = np.full((input_frame.shape[0] + 2, input_frame.shape[1] + 2), pad_value, dtype=input_frame.dtype)
    input_frame_indicator[0, 0] = indicator_value
    input_frame_indicator[-1, -1] = indicator_value
    input_frame_indicator[1:-1, 1:-1] = input_frame
    output_frame_indicator = np.full((output_frame.shape[0] + 2, output_frame.shape[1] + 2), pad_value, dtype=output_frame.dtype)
    output_frame_indicator[0, 0] = indicator_value
    output_frame_indicator[-1, -1] = indicator_value
    output_frame_indicator[1:-1, 1:-1] = output_frame
    input_frame = input_frame_indicator
    output_frame = output_frame_indicator

    inp_h, inp_w = input_frame.shape
    out_h, out_w = output_frame.shape

    padded_input = np.full((target_size, target_size), pad_value, dtype=input_frame.dtype)
    padded_output = np.full((target_size, target_size), pad_value, dtype=output_frame.dtype)

    max_h_offset = target_size - max(inp_h, out_h)
    max_w_offset = target_size - max(inp_w, out_w)
    random_h_offset = np.random.randint(0, max_h_offset + 1) if max_h_offset > 0 else 0
    random_w_offset = np.random.randint(0, max_w_offset + 1) if max_w_offset > 0 else 0

    padded_input[random_h_offset:random_h_offset + inp_h, random_w_offset:random_w_offset + inp_w] = input_frame
    padded_output[random_h_offset:random_h_offset + out_h, random_w_offset:random_w_offset + out_w] = output_frame

    return padded_input, padded_output


class ArcTask:
    """Represents a single ARC task with multiple input-output examples."""
    def __init__(self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]):
        self.task_id = task_id
        self.examples = examples  # List of (input_grid, output_grid) tuples
        
    def __len__(self):
        return len(self.examples)
    
    def sample_pair(self, replace=False) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Sample 2 examples from this task."""
        if len(self.examples) < 2:
            # If only 1 example, return it twice
            return self.examples[0], self.examples[0]
        
        indices = np.random.choice(len(self.examples), size=2, replace=replace)
        return self.examples[indices[0]], self.examples[indices[1]]


class ArcTaskDataset(Dataset):
    """
    Task-based ARC Dataset. Each item is a complete task with all its examples.
    Much cleaner design for task-level operations.
    """
    def __init__(self, train_path: str, sol_path: str, min_examples: int = 2):
        """
        Args:
            train_path: Path to training JSON file
            sol_path: Path to solutions JSON file
            min_examples: Minimum number of examples required per task (default: 2)
        """
        train_data = json.load(open(train_path))
        sol_data = json.load(open(sol_path))
        
        self.tasks: List[ArcTask] = []
        
        for task_id in train_data:
            examples = []
            puzzle = train_data[task_id]
            
            # Add training examples
            for example in puzzle['train']:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                examples.append((input_grid, output_grid))
            
            # Add test example with solution
            input_grid = np.array(puzzle['test'][0]['input'])
            output_grid = np.array(sol_data[task_id][0])
            examples.append((input_grid, output_grid))
            
            # Only include tasks with sufficient examples
            if len(examples) >= min_examples:
                self.tasks.append(ArcTask(task_id, examples))
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> ArcTask:
        """Return the entire task object."""
        return self.tasks[idx]


def collate_task_pairs(batch: List[ArcTask], target_size: int = 32):
    """
    Collate function for task-based dataset that creates pairs of pairs.
    
    Args:
        batch: List of ArcTask objects
        target_size: Target grid size for padding (default: 32)
    
    Returns:
        tuple: (task_ids, data) where:
            - task_ids: List of task IDs (length N)
            - data: np.array of shape (N, 2, 32, 32, 2)
    """
    batch_data = []
    task_ids = []
    
    for task in batch:
        # Sample 2 examples from this task
        example_1, example_2 = task.sample_pair(replace=False)
        
        # Generate single augmentation for this task
        perm = np.random.permutation(10)
        aug = np.random.randint(0, 8)
        
        # Apply same augmentation to both examples
        augmented_examples = []
        for input_grid, output_grid in [example_1, example_2]:
            # Color permutation
            input_perm = color_permutation(input_grid, perm)
            output_perm = color_permutation(output_grid, perm)
            
            # Dihedral transform
            input_aug = dihedral_transform(input_perm, aug)
            output_aug = dihedral_transform(output_perm, aug)
            
            # Pad and stack (add 2 to shift colors for padding)
            padded_input, padded_output = pad_frames(
                input_aug + 2, output_aug + 2,
                target_size=target_size, pad_value=0, indicator_value=1
            )
            
            # Stack input and output as channels: shape (32, 32, 2)
            stacked = np.stack([padded_input, padded_output], axis=-1)
            augmented_examples.append(stacked)
        
        # Stack the two examples: shape (2, 32, 32, 2)
        batch_element = np.stack(augmented_examples, axis=0)
        batch_data.append(batch_element)
        task_ids.append(task.task_id)
    
    # Stack all batch elements: shape (N, 2, 32, 32, 2)
    return task_ids, np.stack(batch_data, axis=0)


# Convenience function for single-example collation (backward compatibility)
def collate_task_single(batch: List[ArcTask], target_size: int = 32):
    """
    Collate function that returns single examples from each task.
    
    Returns:
        tuple: (task_ids, inputs, outputs) where each has shape (N, 32, 32)
    """
    task_ids = []
    inputs = []
    outputs = []
    
    for task in batch:
        # Sample one example
        idx = np.random.randint(0, len(task.examples))
        input_grid, output_grid = task.examples[idx]
        
        # Apply augmentation
        perm = np.random.permutation(10)
        aug = np.random.randint(0, 8)
        
        input_perm = color_permutation(input_grid, perm)
        output_perm = color_permutation(output_grid, perm)
        
        input_aug = dihedral_transform(input_perm, aug)
        output_aug = dihedral_transform(output_perm, aug)
        
        # Pad
        padded_input, padded_output = pad_frames(
            input_aug + 2, output_aug + 2,
            target_size=target_size, pad_value=0, indicator_value=1
        )
        
        task_ids.append(task.task_id)
        inputs.append(padded_input)
        outputs.append(padded_output)
    
    return task_ids, np.array(inputs), np.array(outputs)
