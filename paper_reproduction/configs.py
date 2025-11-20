"""
Example Configuration for Paper Reproduction
============================================

This file contains example configurations for different use cases.
"""

# Quick test configuration (for debugging)
QUICK_TEST_CONFIG = {
    'num_colors': 10,
    'base_channels': 32,
    'latent_channels': 256,
    'n_res_blocks': 1,
    
    'batch_size': 8,
    'epochs': 5,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'num_workers': 2,
    
    'use_memory_efficient': True,
    'n_inner_steps': 2,
    'n_outer_steps': 2,
    'eval_n_steps': 4,
    
    'use_ema': False,
    'use_wandb': False,
    
    'subset_size': 100,  # Use small subset
}

# Standard configuration (balanced performance/speed)
STANDARD_CONFIG = {
    'num_colors': 10,
    'base_channels': 64,
    'latent_channels': 512,
    'n_res_blocks': 2,
    
    'batch_size': 16,
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,
    
    'use_memory_efficient': True,
    'n_inner_steps': 3,
    'n_outer_steps': 3,
    'eval_n_steps': 9,
    'eval_use_deep_recursion': True,
    
    'use_ema': True,
    'ema_decay': 0.9995,
    'use_wandb': False,
    
    'subset_size': None,
}

# High performance configuration (best quality, more resources)
HIGH_PERFORMANCE_CONFIG = {
    'num_colors': 10,
    'base_channels': 96,
    'latent_channels': 768,
    'n_res_blocks': 3,
    
    'batch_size': 8,  # Smaller batch for larger model
    'epochs': 100,
    'lr': 5e-5,
    'weight_decay': 1e-5,
    'num_workers': 6,
    
    'use_memory_efficient': True,
    'n_inner_steps': 4,
    'n_outer_steps': 4,
    'eval_n_steps': 16,
    'eval_use_deep_recursion': True,
    
    'use_ema': True,
    'ema_decay': 0.9995,
    'use_wandb': True,
    'project_name': 'arc-prize-deep-recursion-hp',
    
    'subset_size': None,
}

# Memory-constrained configuration (for smaller GPUs)
MEMORY_CONSTRAINED_CONFIG = {
    'num_colors': 10,
    'base_channels': 48,
    'latent_channels': 384,
    'n_res_blocks': 2,
    
    'batch_size': 8,
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 2,
    
    'use_memory_efficient': True,
    'n_inner_steps': 2,  # Fewer inner steps
    'n_outer_steps': 5,  # More outer steps (same total)
    'eval_n_steps': 10,
    'eval_use_deep_recursion': True,
    
    'use_ema': True,
    'ema_decay': 0.9995,
    'use_wandb': False,
    
    'subset_size': None,
}

# Ablation: Standard training (no memory-efficient mode)
STANDARD_TRAINING_CONFIG = {
    'num_colors': 10,
    'base_channels': 64,
    'latent_channels': 512,
    'n_res_blocks': 2,
    
    'batch_size': 16,
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,
    
    'use_memory_efficient': False,  # Standard training
    'n_steps': 6,  # Total steps
    'eval_n_steps': 9,
    
    'use_ema': True,
    'ema_decay': 0.9995,
    'use_wandb': False,
    
    'subset_size': None,
}


def get_config(name='standard'):
    """
    Get a configuration by name.
    
    Args:
        name: Configuration name ('quick_test', 'standard', 'high_performance', 
              'memory_constrained', 'standard_training')
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'quick_test': QUICK_TEST_CONFIG,
        'standard': STANDARD_CONFIG,
        'high_performance': HIGH_PERFORMANCE_CONFIG,
        'memory_constrained': MEMORY_CONSTRAINED_CONFIG,
        'standard_training': STANDARD_TRAINING_CONFIG,
    }
    
    if name not in configs:
        raise ValueError(f"Unknown configuration: {name}. Available: {list(configs.keys())}")
    
    config = configs[name].copy()
    
    # Add default paths
    config.setdefault('data_path', 'data')
    config.setdefault('checkpoint_path', f'paper_reproduction/checkpoints/{name}_best_model.pt')
    config.setdefault('device', 'cuda')
    
    return config


if __name__ == "__main__":
    # Print all configurations
    print("Available Configurations:")
    print("=" * 70)
    
    for name in ['quick_test', 'standard', 'high_performance', 'memory_constrained', 'standard_training']:
        config = get_config(name)
        print(f"\n{name.upper().replace('_', ' ')}:")
        print("-" * 70)
        
        # Model info
        print(f"  Model: {config['base_channels']} base, {config['latent_channels']} latent, {config['n_res_blocks']} res blocks")
        
        # Training info
        print(f"  Training: batch={config['batch_size']}, lr={config['lr']}, epochs={config['epochs']}")
        
        # Recursion info
        if config.get('use_memory_efficient'):
            total = config['n_inner_steps'] * config['n_outer_steps']
            print(f"  Recursion: {config['n_inner_steps']} inner × {config['n_outer_steps']} outer = {total} steps (memory-efficient)")
        else:
            print(f"  Recursion: {config['n_steps']} steps (standard)")
        
        # Other
        print(f"  EMA: {config.get('use_ema', False)}, WandB: {config.get('use_wandb', False)}")
        
        if config.get('subset_size'):
            print(f"  Data: subset of {config['subset_size']} samples")
