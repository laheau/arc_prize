import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os

# Reuse some logic from existing dataset code if possible, but keep it self-contained for VARC reproduction
def pad_grid(grid, target_size=32, pad_value=0):
    h, w = grid.shape
    if h > target_size or w > target_size:
        # Resize or crop? ARC grids are usually small. 
        # If larger, we might need to crop or scale. For now assume they fit or crop.
        # In ARC, grids > 30 are rare.
        h = min(h, target_size)
        w = min(w, target_size)
        grid = grid[:h, :w]
        
    padded = np.full((target_size, target_size), pad_value, dtype=grid.dtype)
    # Center crop or top-left? Top-left is standard for ARC usually.
    # Random placement is better for training (augmentation).
    
    # For deterministic evaluation, use top-left or center.
    # Let's use random for training, top-left for eval.
    return padded, h, w

def random_pad_grid(grid, target_size=32, pad_value=0):
    h, w = grid.shape
    h = min(h, target_size)
    w = min(w, target_size)
    grid = grid[:h, :w]
    
    padded = np.full((target_size, target_size), pad_value, dtype=grid.dtype)
    
    max_h_off = target_size - h
    max_w_off = target_size - w
    
    off_h = np.random.randint(0, max_h_off + 1)
    off_w = np.random.randint(0, max_w_off + 1)
    
    padded[off_h:off_h+h, off_w:off_w+w] = grid
    return padded

class VARCDataset(Dataset):
    def __init__(self, data_path, mode='train', target_size=32, augment=True):
        """
        mode: 'train' or 'test'
        """
        self.data = json.load(open(data_path))
        self.task_ids = list(self.data.keys())
        self.mode = mode
        self.target_size = target_size
        self.augment = augment
        
        # Flatten all examples into a list of (input, output) pairs for training
        self.examples = []
        if mode == 'train':
            for tid in self.task_ids:
                task = self.data[tid]
                for ex in task['train']:
                    self.examples.append({
                        'input': np.array(ex['input']),
                        'output': np.array(ex['output']),
                        'task_id': tid
                    })
                # We can also use test pairs if we have solutions, but usually 'train' mode implies training on train pairs
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.examples)
        else:
            return len(self.task_ids)
            
    def __getitem__(self, idx):
        if self.mode == 'train':
            ex = self.examples[idx]
            inp = ex['input']
            out = ex['output']
            
            if self.augment:
                # Augmentations: Dihedral, Color Permutation
                # 1. Dihedral
                k = np.random.randint(0, 8)
                inp = self.dihedral_transform(inp, k)
                out = self.dihedral_transform(out, k)
                
                # 2. Color Permutation (0 is usually background, keep it? Paper says "solely on ARC data", usually permutation is safe)
                # Some papers keep 0 fixed, some permute all. Let's permute 1-9, keep 0 fixed often.
                # Or permute all.
                perm = np.random.permutation(10)
                inp = self.color_permute(inp, perm)
                out = self.color_permute(out, perm)
                
                # 3. Random Padding
                inp_padded = random_pad_grid(inp, self.target_size, pad_value=0)
                out_padded = random_pad_grid(out, self.target_size, pad_value=0)
            else:
                # Center or Top-Left
                inp_padded, _, _ = self.pad_grid_fixed(inp, self.target_size)
                out_padded, _, _ = self.pad_grid_fixed(out, self.target_size)
                
            return {
                'input': torch.LongTensor(inp_padded),
                'output': torch.LongTensor(out_padded),
                'task_id': ex['task_id']
            }
            
        else:
            # Test mode: Return the whole task for TTT
            tid = self.task_ids[idx]
            task = self.data[tid]
            return {
                'task_id': tid,
                'train': task['train'],
                'test': task['test']
            }

    def dihedral_transform(self, arr, k):
        if k == 0: return arr
        if k == 1: return np.rot90(arr, 1)
        if k == 2: return np.rot90(arr, 2)
        if k == 3: return np.rot90(arr, 3)
        if k == 4: return np.fliplr(arr)
        if k == 5: return np.flipud(arr)
        if k == 6: return arr.T
        if k == 7: return np.fliplr(np.rot90(arr, 1))
        return arr

    def color_permute(self, arr, perm):
        return perm[arr]

    def pad_grid_fixed(self, grid, target_size=32, pad_value=0):
        h, w = grid.shape
        h = min(h, target_size)
        w = min(w, target_size)
        grid = grid[:h, :w]
        padded = np.full((target_size, target_size), pad_value, dtype=grid.dtype)
        padded[:h, :w] = grid
        return padded, h, w

def collate_fn(batch):
    # For training
    inputs = torch.stack([b['input'] for b in batch])
    outputs = torch.stack([b['output'] for b in batch])
    return inputs, outputs
