# Quick Reference Card - Paper Reproduction

## One-Line Summary
Deep recursive model with gradient detachment for memory-efficient training on ARC tasks.

## Quick Start

```bash
# View examples
python paper_reproduction/examples.py

# Run tests
python paper_reproduction/test.py

# Start training
python paper_reproduction/main.py
```

## Core API

### Model Creation
```python
from paper_reproduction.model import ARCDeepRecursiveModel

model = ARCDeepRecursiveModel(
    num_colors=10,           # Number of colors in puzzles
    base_channels=64,        # Base feature channels
    latent_channels=512,     # Latent space size
    n_res_blocks=2          # Residual blocks per stage
)
```

### Forward Pass
```python
# Standard forward (keeps full graph)
logits = model(input, n_steps=3)

# Memory-efficient forward (gradient detachment)
logits, all_outputs = model.deep_recursion_forward(
    input,
    n_inner_steps=3,    # Fast inner iterations
    n_outer_steps=3,    # Checkpointed outer steps
    detach_outer=True   # Detach gradients between outer steps
)
```

### Training
```python
from paper_reproduction.train import train
from paper_reproduction.configs import get_config

# Get configuration
config = get_config('standard')  # or 'quick_test', 'high_performance', etc.

# Train
best_acc = train(model, train_loader, val_loader, config)
```

## Key Parameters

### Recursion Configuration
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_inner_steps` | Inner recursion steps | 2-4 |
| `n_outer_steps` | Outer steps (checkpointed) | 2-5 |
| **Total steps** | `inner × outer` | 6-16 |

### Model Size
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `base_channels` | Feature channels | 32, 64, 96 |
| `latent_channels` | Latent size | 256, 512, 768 |
| `n_res_blocks` | Depth | 1-3 |

### Training
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `batch_size` | Batch size | 8-32 |
| `lr` | Learning rate | 5e-5 to 1e-4 |
| `use_ema` | Use EMA | True (recommended) |
| `use_memory_efficient` | Memory-efficient mode | True (recommended) |

## Configurations Cheat Sheet

```python
from paper_reproduction.configs import get_config

# Quick test (5 min, 100 samples)
config = get_config('quick_test')

# Standard (recommended, ~8 hours on GPU)
config = get_config('standard')

# High performance (best quality, ~16 hours)
config = get_config('high_performance')

# Memory constrained (for 4GB GPUs)
config = get_config('memory_constrained')

# Standard training (non-memory-efficient baseline)
config = get_config('standard_training')
```

## Memory Usage Guide

| Configuration | Batch Size | VRAM Usage |
|---------------|-----------|------------|
| Small model, memory-efficient | 16 | ~2 GB |
| Standard model, memory-efficient | 16 | ~4 GB |
| Standard model, standard training | 16 | ~8 GB |
| Large model, memory-efficient | 8 | ~6 GB |

## Common Patterns

### Training from Scratch
```python
from paper_reproduction.main import main
main()  # Uses default config
```

### Custom Training Loop
```python
from paper_reproduction.train import train_step_memory_efficient

for epoch in range(epochs):
    for batch in dataloader:
        loss, metrics = train_step_memory_efficient(
            model, batch, optimizer,
            n_inner_steps=3, n_outer_steps=3
        )
```

### Inference
```python
model.eval()
with torch.no_grad():
    # Use more steps for better quality
    logits = model(input, n_steps=16)
    predictions = logits.argmax(dim=-1)
```

### Loading Checkpoint
```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` or use `memory_constrained` config |
| Training too slow | Reduce `n_res_blocks`, increase `batch_size` |
| Poor performance | Increase `eval_n_steps`, enable EMA, train longer |
| Unstable training | Enable EMA, reduce learning rate |

## File Map

| File | Purpose |
|------|---------|
| `model.py` | Model implementation |
| `train.py` | Training utilities |
| `main.py` | Main training script |
| `configs.py` | Preset configurations |
| `test.py` | Test suite |
| `examples.py` | Usage examples |
| `README.md` | Full documentation |
| `SUMMARY.md` | Implementation overview |
| `COMPARISON.md` | vs existing code |

## Key Techniques

1. **Gradient Detachment**: `detach_outer=True` in `deep_recursion_forward()`
2. **Task + Output Summing**: Concatenate task features and current output
3. **Trajectory Rollout**: Multiple outer steps with gradient accumulation
4. **EMA**: `use_ema=True` for training stability

## Performance Metrics

```python
# After training
print(f"Cell accuracy: {metrics['cell_accuracy']:.4f}")
print(f"Puzzle accuracy: {metrics['puzzle_accuracy']:.4f}")
```

## Next Steps

1. Run tests: `python paper_reproduction/test.py`
2. Try quick test: Modify `main.py` to use `quick_test` config
3. Full training: Run `python paper_reproduction/main.py`
4. Tune hyperparameters using `configs.py`
5. Enable wandb logging: Set `use_wandb=True`

## Resources

- **Documentation**: `README.md`, `SUMMARY.md`
- **Examples**: `examples.py`
- **Tests**: `test.py`
- **Paper**: arXiv:2511.14761
