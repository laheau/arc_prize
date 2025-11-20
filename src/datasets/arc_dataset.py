
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


class ArcPuzzle:
    def __init__(self, id: str, examples: List[Tuple[np.ndarray, np.ndarray]]):
        self.id = id
        self.examples = examples

class ArcDataset(Dataset):
    def __init__(self, train_path: str, sol_path: str):
        train_data = json.load(open(train_path))
        sol_data = json.load(open(sol_path))
        self.data : List[ArcPuzzle] = []
        self.len = 0
        self.index_to_puzzle : List[int] = []
        self.prefix_sum : List[int] = [0]
        for id in train_data:
            arc_puzzle = []
            puzzle = train_data[id]
            for example in puzzle['train']:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                arc_puzzle.append((input_grid, output_grid))
                self.index_to_puzzle.append(len(self.data))
                
                self.len += 1
            input_grid = np.array(puzzle['test'][0]['input'])
            output_grid = np.array(sol_data[id][0])
            arc_puzzle.append((input_grid, output_grid))
            self.index_to_puzzle.append(len(self.data))
            self.len += 1
            self.data.append(ArcPuzzle(id, arc_puzzle))
            self.prefix_sum.append(self.len)



    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Any:
        item = self.data[self.index_to_puzzle[idx]].examples[idx - self.prefix_sum[self.index_to_puzzle[idx]]]
        id = self.data[self.index_to_puzzle[idx]].id
        return (id, item)




def collate_fn(batch):
    ids, items = zip(*batch)
    permutations = [np.random.permutation(10) for _ in range(len(items))]
    permutated_data = [(color_permutation(data[0], perm), color_permutation(data[1], perm)) for data, perm in zip(items, permutations)]
    augmentations = [np.random.randint(0, 8) for _ in range(len(permutated_data))]
    transformed_data = [(dihedral_transform(data[0], aug), dihedral_transform(data[1], aug)) for data, aug in zip(permutated_data, augmentations)]
    padded_data = [pad_frames(data[0] + 2, data[1] + 2, target_size=32, pad_value=0, indicator_value=1) for data in transformed_data]
    return ids, np.array([data[0] for data in padded_data]), np.array([data[1] for data in padded_data])



def pad_frames(input_frame, output_frame, target_size=32, pad_value=0, indicator_value=1):
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


def collate_fn_pairs(batch):
    """
    Collate function that generates pairs of pairs from the same task_id.
    For each batch element, samples 2 examples from the same task and applies
    the same augmentation to both.
    
    Args:
        batch: List of (id, (input_grid, output_grid)) tuples
        
    Returns:
        tuple: (ids, tensor) where:
            - ids: List of task IDs (length N)
            - tensor: np.array of shape (N, 2, 32, 32, 2) where:
                - N is batch size
                - 2 is the number of example pairs per batch element
                - 32x32 is the grid size
                - 2 is input/output channels
    """
    ids, items = zip(*batch)
    
    # Group items by task_id
    task_groups = {}
    for task_id, item in zip(ids, items):
        if task_id not in task_groups:
            task_groups[task_id] = []
        task_groups[task_id].append(item)
    
    batch_data = []
    batch_ids = []
    
    for task_id, examples in task_groups.items():
        # If we have at least 2 examples from this task, create pairs
        if len(examples) >= 2:
            # Sample 2 different examples from this task
            indices = np.random.choice(len(examples), size=2, replace=False)
            pair_1 = examples[indices[0]]
            pair_2 = examples[indices[1]]
            
            # Generate single augmentation for this batch element
            perm = np.random.permutation(10)
            aug = np.random.randint(0, 8)
            
            # Apply same augmentation to both pairs
            augmented_pairs = []
            for pair in [pair_1, pair_2]:
                # Color permutation
                input_perm = color_permutation(pair[0], perm)
                output_perm = color_permutation(pair[1], perm)
                
                # Dihedral transform
                input_aug = dihedral_transform(input_perm, aug)
                output_aug = dihedral_transform(output_perm, aug)
                
                # Pad and stack (add 2 to shift colors for padding)
                padded_input, padded_output = pad_frames(
                    input_aug + 2, output_aug + 2, 
                    target_size=32, pad_value=0, indicator_value=1
                )
                
                # Stack input and output as channels: shape (32, 32, 2)
                stacked = np.stack([padded_input, padded_output], axis=-1)
                augmented_pairs.append(stacked)
            
            # Stack the two pairs: shape (2, 32, 32, 2)
            batch_element = np.stack(augmented_pairs, axis=0)
            batch_data.append(batch_element)
            batch_ids.append(task_id)
    
    if len(batch_data) == 0:
        # Fallback if no valid pairs found
        return [], np.array([])
    
    # Stack all batch elements: shape (N, 2, 32, 32, 2)
    return batch_ids, np.stack(batch_data, axis=0)


def get_task_sampler(dataset: ArcDataset):
    """
    Create a custom sampler that ensures we sample from tasks that have
    at least 2 examples (needed for collate_fn_pairs).
    
    Args:
        dataset: ArcDataset instance
        
    Returns:
        List of valid indices that can be used with collate_fn_pairs
    """
    from collections import defaultdict
    
    # Group indices by task_id
    task_indices = defaultdict(list)
    for idx in range(len(dataset)):
        task_id, _ = dataset[idx]
        task_indices[task_id].append(idx)
    
    # Keep only tasks with at least 2 examples
    valid_indices = []
    for task_id, indices in task_indices.items():
        if len(indices) >= 2:
            valid_indices.extend(indices)
    
    return valid_indices