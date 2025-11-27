"""
Main script for training the Deep Recursive Model on ARC tasks
===============================================================

This script demonstrates how to use the paper reproduction implementation
with the existing ARC dataset infrastructure.
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader, Subset
from paper_reproduction.model import ARCDeepRecursiveModel, count_parameters
from paper_reproduction.train import train
from src.datasets.arc_dataset import ArcDataset, collate_fn


def create_arc_dataloader(data_path, batch_size=16, num_workers=4, subset_size=None):
    """
    Create a dataloader for ARC dataset.
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        subset_size: If specified, use only a subset of the data
        
    Returns:
        train_loader, val_loader
    """
    # Load datasets
    train_dataset = ArcDataset(
        train_path=os.path.join(data_path, 'arc-agi_training_challenges.json'),
        sol_path=os.path.join(data_path, 'arc-agi_training_solutions.json')
    )
    
    val_dataset = ArcDataset(
        train_path=os.path.join(data_path, 'arc-agi_evaluation_challenges.json'),
        sol_path=os.path.join(data_path, 'arc-agi_evaluation_solutions.json')
    )
    
    # Use subset if specified
    if subset_size is not None:
        train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(subset_size // 2, len(val_dataset))))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def collate_wrapper(batch):
    """Wrapper for collate_fn to match expected format."""
    ids, inputs, outputs = collate_fn(batch)
    # Convert to tensors
    inputs = torch.from_numpy(inputs)
    outputs = torch.from_numpy(outputs)
    return inputs, outputs


def main():
    """Main training function."""
    
    # Configuration
    config = {
        # Model configuration
        'num_colors': 10,
        'base_channels': 64,
        'latent_channels': 512,
        'n_res_blocks': 2,
        
        # Training configuration
        'batch_size': 16,
        'epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        
        # Recursion configuration
        'use_memory_efficient': True,
        'n_inner_steps': 3,  # Number of inner recursion steps per outer step
        'n_outer_steps': 3,  # Number of outer steps (with gradient detachment)
        'n_steps': 3,  # For standard training (if not using memory efficient)
        'eval_n_steps': 9,  # More steps during evaluation for better performance
        'eval_use_deep_recursion': True,
        
        # EMA configuration
        'use_ema': True,
        'ema_decay': 0.9995,
        
        # Logging
        'use_wandb': False,  # Set to True to enable wandb logging
        'project_name': 'arc-prize-deep-recursion',
        
        # Paths
        'data_path': 'data',
        'checkpoint_path': 'paper_reproduction/checkpoints/best_model.pt',
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Debugging (set to a small number to quickly test the pipeline)
        'subset_size': None,  # Set to e.g., 100 for quick testing
    }
    
    print("="*70)
    print("Deep Recursive Model for ARC Prize - Paper Reproduction")
    print("="*70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n" + "="*70)
    
    # Initialize wandb if enabled
    if config['use_wandb']:
        import wandb
        wandb.init(
            project=config['project_name'],
            config=config,
            name=f"deep-rec-inner{config['n_inner_steps']}-outer{config['n_outer_steps']}"
        )
    
    # Create model
    print("\nCreating model...")
    model = ARCDeepRecursiveModel(
        num_colors=config['num_colors'],
        base_channels=config['base_channels'],
        latent_channels=config['latent_channels'],
        n_res_blocks=config['n_res_blocks']
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(config['checkpoint_path']), exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    
    # Check if data path exists
    data_path = config['data_path']
    if not os.path.exists(data_path):
        print(f"ERROR: Data path '{data_path}' does not exist!")
        print("Please ensure the ARC dataset is available.")
        print("You can download it from: https://github.com/fchollet/ARC-AGI")
        return
    
    try:
        # Create custom dataloader wrapper
        from torch.utils.data import Dataset as TorchDataset
        
        class ARCDatasetWrapper(TorchDataset):
            def __init__(self, arc_dataset):
                self.dataset = arc_dataset
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                return self.dataset[idx]
        
        # Load datasets
        train_dataset = ArcDataset(
            train_path=os.path.join(data_path, 'arc-agi_training_challenges.json'),
            sol_path=os.path.join(data_path, 'arc-agi_training_solutions.json')
        )
        
        val_dataset = ArcDataset(
            train_path=os.path.join(data_path, 'arc-agi_evaluation_challenges.json'),
            sol_path=os.path.join(data_path, 'arc-agi_evaluation_solutions.json')
        )
        
        # Use subset if specified
        if config['subset_size'] is not None:
            train_dataset = Subset(train_dataset, range(min(config['subset_size'], len(train_dataset))))
            val_dataset = Subset(val_dataset, range(min(config['subset_size'] // 2, len(val_dataset))))
        
        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_wrapper,
            num_workers=config['num_workers'],
            pin_memory=True if config['device'] == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'] * 2,
            shuffle=False,
            collate_fn=collate_wrapper,
            num_workers=config['num_workers'],
            pin_memory=True if config['device'] == 'cuda' else False
        )
        
    except Exception as e:
        print(f"ERROR: Failed to load datasets: {e}")
        print("\nPlease ensure the ARC dataset files are in the correct location:")
        print(f"  - {data_path}/arc-agi_training_challenges.json")
        print(f"  - {data_path}/arc-agi_training_solutions.json")
        print(f"  - {data_path}/arc-agi_evaluation_challenges.json")
        print(f"  - {data_path}/arc-agi_evaluation_solutions.json")
        return
    
    # Train
    print("\nStarting training...")
    print(f"Using {'memory-efficient' if config['use_memory_efficient'] else 'standard'} training")
    print(f"Device: {config['device']}")
    print("\n" + "="*70 + "\n")
    
    try:
        best_acc = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        print("\n" + "="*70)
        print(f"Training completed successfully!")
        print(f"Best validation puzzle accuracy: {best_acc:.4f}")
        print("="*70)
        
        if config['use_wandb']:
            wandb.finish()
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        if config['use_wandb']:
            wandb.finish()
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        if config['use_wandb']:
            wandb.finish()


if __name__ == "__main__":
    main()
