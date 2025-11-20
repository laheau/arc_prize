# Paper Reproduction: Deep Recursive Model for ARC Prize

This folder contains an implementation of key techniques from the research paper for solving ARC (Abstraction and Reasoning Corpus) tasks.

## Overview

This implementation focuses on the most effective techniques from the paper, specifically:

### 1. **Deep Recursion with Gradient Detachment**
- The model performs multiple recursive refinement steps
- Gradients are detached between outer steps to reduce memory usage
- Inner steps perform latent-space recursion for efficient computation

### 2. **Task + Output Vector Summing**
- The model takes both the task input (x) and current output (y) as input
- These are projected and combined before processing
- This allows the model to refine its output iteratively

### 3. **Memory-Efficient Training**
- Implements gradient accumulation at intermediate recursion steps
- Avoids keeping the full computational graph in memory
- Significantly reduces VRAM usage while maintaining training quality

### 4. **Trajectory Rollout**
- Multiple gradient updates through recursive steps
- Each outer step accumulates gradients that are then applied
- Enables deep recursion without excessive memory requirements

## Architecture

The model consists of:

- **Encoder**: Convolutional encoder with residual blocks that processes concatenated task and output features
- **Decoder**: U-Net style decoder with skip connections for reconstruction
- **Recursive Processing**: 
  - Inner loop: Latent-space recursion (fast iterations)
  - Outer loop: Full forward passes with gradient detachment (memory-efficient)

## Files

- `model.py`: Core model implementation
  - `DeepRecursiveModel`: Main recursive model
  - `ARCDeepRecursiveModel`: Wrapper for ARC tasks with one-hot encoding
  - `Encoder` and `Decoder`: Network components

- `train.py`: Training utilities
  - `train_step_memory_efficient`: Memory-efficient training with gradient accumulation
  - `train_step_standard`: Standard training (for comparison)
  - `train`: Main training loop with EMA support

- `main.py`: Complete training script
  - Dataset loading
  - Configuration
  - Training orchestration

## Usage

### Basic Training

```python
from paper_reproduction.main import main

# Run with default configuration
main()
```

### Custom Configuration

```python
from paper_reproduction.model import ARCDeepRecursiveModel
from paper_reproduction.train import train

# Create model
model = ARCDeepRecursiveModel(
    num_colors=10,
    base_channels=64,
    latent_channels=512,
    n_res_blocks=2
)

# Configure training
config = {
    'batch_size': 16,
    'epochs': 50,
    'lr': 1e-4,
    'use_memory_efficient': True,
    'n_inner_steps': 3,  # Inner recursion steps
    'n_outer_steps': 3,  # Outer steps with gradient detachment
    'use_ema': True,
    'ema_decay': 0.9995,
    'device': 'cuda'
}

# Train
train(model, train_loader, val_loader, config)
```

### Running from Command Line

```bash
# Basic training
cd paper_reproduction
python main.py

# Or from the repository root
python -m paper_reproduction.main
```

## Key Techniques Explained

### 1. Memory-Efficient Gradient Accumulation

Instead of keeping the full computational graph for all recursion steps:

```python
# Standard approach (high memory):
for step in range(n_steps):
    output = model(input, output)  # Graph keeps growing
loss = compute_loss(output, target)
loss.backward()  # Backprop through entire graph

# Memory-efficient approach:
for step in range(n_outer_steps):
    # Inner steps with detached gradients
    for inner_step in range(n_inner_steps):
        output = model(input, output)
    
    # Compute loss and gradients for this outer step
    step_loss = compute_loss(output, target)
    step_loss.backward()  # Gradients accumulate
    
    # Detach for next outer step (crucial!)
    output = output.detach()
```

### 2. Task + Output Summing

The model processes both task input and current output:

```python
# Project inputs
task_feat = task_proj(task_input)
output_feat = output_proj(current_output)

# Combine (concatenate or sum)
combined = task_feat + output_feat

# Process through network
new_output = network(combined)
```

### 3. Deep Recursion Configuration

The model supports different recursion strategies:

- **Standard recursion**: `model(input, n_steps=N)`
  - Simple N-step recursion
  - Higher memory usage
  
- **Deep recursion**: `model.deep_recursion_forward(input, n_inner_steps=3, n_outer_steps=3)`
  - 3 outer steps, each with 3 inner steps
  - Gradient detachment between outer steps
  - Much lower memory usage

## Configuration Options

### Model Parameters
- `num_colors`: Number of colors in ARC puzzles (default: 10)
- `base_channels`: Base number of feature channels (default: 64)
- `latent_channels`: Latent space dimensions (default: 512)
- `n_res_blocks`: Residual blocks per stage (default: 2)

### Training Parameters
- `batch_size`: Batch size (default: 16)
- `lr`: Learning rate (default: 1e-4)
- `weight_decay`: Weight decay for regularization (default: 1e-5)
- `epochs`: Number of training epochs (default: 50)

### Recursion Parameters
- `use_memory_efficient`: Use memory-efficient training (default: True)
- `n_inner_steps`: Inner recursion steps (default: 3)
- `n_outer_steps`: Outer recursion steps (default: 3)
- `eval_n_steps`: Steps during evaluation (default: 9)

### Optimization Parameters
- `use_ema`: Use Exponential Moving Average (default: True)
- `ema_decay`: EMA decay rate (default: 0.9995)

## Performance Tips

1. **Memory vs. Quality Trade-off**
   - More outer steps = more memory, better quality
   - More inner steps = less memory per outer step
   - Typical good config: 3 inner × 3 outer = 9 total steps

2. **Evaluation**
   - Use more steps during evaluation: `eval_n_steps=9` or higher
   - The model can refine outputs with more steps at test time

3. **Training Stability**
   - EMA helps with stability
   - Gradient clipping is applied (max_norm=1.0)
   - Loss is averaged over all intermediate steps

## Requirements

- PyTorch >= 2.0
- numpy
- tqdm
- wandb (optional, for logging)

## Model Size

With default parameters:
- Parameters: ~15-20M (depending on configuration)
- VRAM usage (training):
  - Memory-efficient mode: ~2-4 GB (batch_size=16)
  - Standard mode: ~6-10 GB (batch_size=16)

## Techniques NOT Included

This implementation focuses on the working techniques. The following were tested in the paper but found less effective:

- Various ablation study configurations
- Alternative loss formulations that didn't improve results
- Techniques that required significantly more compute without clear benefits

## Citation

If you use this implementation, please cite the original paper:
```
@article{paper2025,
  title={Paper Title},
  author={Authors},
  journal={arXiv preprint arXiv:2511.14761},
  year={2025}
}
```

## License

This implementation is provided for research and educational purposes.
