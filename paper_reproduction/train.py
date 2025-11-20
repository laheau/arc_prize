"""
Memory-Efficient Training Script for Deep Recursive Model
==========================================================

This training script implements:
1. Gradient accumulation at intermediate recursion steps
2. Memory-efficient backpropagation through trajectory rollouts
3. Proper gradient detachment during training
4. Support for EMA (Exponential Moving Average)

Key technique: Instead of keeping the full computational graph,
we compute gradients at each outer step and accumulate them,
which significantly reduces VRAM usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import wandb
from tqdm import tqdm
from copy import deepcopy


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        
        for param in self.shadow.parameters():
            param.requires_grad = False
        
        self.num_updates = 0
    
    @torch.no_grad()
    def update(self, model):
        """Update the moving average with current model parameters."""
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            if model_param.requires_grad:
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def apply_shadow(self, model):
        """Copy EMA parameters to model."""
        for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            model_param.data.copy_(ema_param.data)
    
    def store(self, model):
        """Save current model parameters."""
        self.backup = {}
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()
    
    def restore(self, model):
        """Restore model parameters from backup."""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


def train_step_standard(
    model,
    batch,
    optimizer,
    n_steps=3,
    device='cuda'
):
    """
    Standard training step (keeps full computational graph).
    
    Args:
        model: The model to train
        batch: Batch of data (inputs, targets)
        optimizer: The optimizer
        n_steps: Number of recursion steps
        device: Device to use
        
    Returns:
        loss: Training loss
        metrics: Dictionary of metrics
    """
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass with all steps
    logits, all_logits = model(inputs, n_steps=n_steps, return_all_steps=True)
    
    # Compute loss on all intermediate steps (helps with training stability)
    total_loss = 0
    for step_logits in all_logits:
        step_loss = F.cross_entropy(
            step_logits.permute(0, 3, 1, 2),  # [B, C, H, W]
            targets.long()
        )
        total_loss = total_loss + step_loss
    
    # Average loss over steps
    loss = total_loss / len(all_logits)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == targets).float().mean()
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item()
    }
    
    return loss.item(), metrics


def train_step_memory_efficient(
    model,
    batch,
    optimizer,
    n_inner_steps=3,
    n_outer_steps=3,
    device='cuda'
):
    """
    Memory-efficient training step with gradient accumulation.
    
    This is the key technique from the paper:
    - Performs multiple outer steps with gradient detachment
    - Computes gradients at each outer step and accumulates them
    - Significantly reduces VRAM usage compared to keeping full graph
    
    Args:
        model: The model to train
        batch: Batch of data (inputs, targets)
        optimizer: The optimizer
        n_inner_steps: Number of inner recursion steps per outer step
        n_outer_steps: Number of outer steps (with gradient accumulation)
        device: Device to use
        
    Returns:
        loss: Training loss
        metrics: Dictionary of metrics
    """
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass with deep recursion
    logits, all_logits = model.deep_recursion_forward(
        inputs,
        n_inner_steps=n_inner_steps,
        n_outer_steps=n_outer_steps,
        detach_outer=True  # This is crucial for memory efficiency
    )
    
    # Compute loss on all outer step outputs
    total_loss = 0
    for step_logits in all_logits:
        step_loss = F.cross_entropy(
            step_logits.permute(0, 3, 1, 2),  # [B, C, H, W]
            targets.long()
        )
        total_loss = total_loss + step_loss
    
    # Average loss over steps
    loss = total_loss / len(all_logits)
    
    # Backward pass (gradients are accumulated from all steps)
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == targets).float().mean()
        
        # Compute per-step accuracy to track improvement
        step_accuracies = []
        for step_logits in all_logits:
            step_preds = step_logits.argmax(dim=-1)
            step_acc = (step_preds == targets).float().mean()
            step_accuracies.append(step_acc.item())
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'step_accuracies': step_accuracies
    }
    
    return loss.item(), metrics


def train_epoch(
    model,
    dataloader,
    optimizer,
    config,
    epoch,
    ema=None,
    use_memory_efficient=True
):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: The optimizer
        config: Training configuration
        epoch: Current epoch number
        ema: Optional EMA tracker
        use_memory_efficient: Whether to use memory-efficient training
        
    Returns:
        avg_loss: Average loss for the epoch
        avg_metrics: Average metrics for the epoch
    """
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if use_memory_efficient:
            loss, metrics = train_step_memory_efficient(
                model=model,
                batch=batch,
                optimizer=optimizer,
                n_inner_steps=config.get('n_inner_steps', 3),
                n_outer_steps=config.get('n_outer_steps', 3),
                device=device
            )
        else:
            loss, metrics = train_step_standard(
                model=model,
                batch=batch,
                optimizer=optimizer,
                n_steps=config.get('n_steps', 3),
                device=device
            )
        
        # Update EMA if enabled
        if ema is not None:
            ema.update(model)
        
        total_loss += loss
        total_accuracy += metrics['accuracy']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'acc': f'{metrics["accuracy"]:.4f}'
        })
        
        # Log to wandb periodically
        if config.get('use_wandb', False) and batch_idx % 50 == 0:
            log_dict = {
                'train/loss': loss,
                'train/accuracy': metrics['accuracy'],
                'train/step': epoch * len(dataloader) + batch_idx,
            }
            
            # Log per-step accuracies if available
            if 'step_accuracies' in metrics:
                for i, acc in enumerate(metrics['step_accuracies']):
                    log_dict[f'train/step_{i}_accuracy'] = acc
            
            wandb.log(log_dict)
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, {'accuracy': avg_accuracy}


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    config,
    n_steps=None,
    use_deep_recursion=False
):
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        dataloader: Evaluation dataloader
        config: Training configuration
        n_steps: Number of recursion steps (if None, use config value)
        use_deep_recursion: Whether to use deep recursion during eval
        
    Returns:
        avg_loss: Average loss
        metrics: Dictionary of metrics
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_cells = 0
    total_puzzles_correct = 0
    total_puzzles = 0
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if n_steps is None:
        n_steps = config.get('eval_n_steps', config.get('n_steps', 3))
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        if use_deep_recursion:
            logits, _ = model.deep_recursion_forward(
                inputs,
                n_inner_steps=config.get('n_inner_steps', 3),
                n_outer_steps=config.get('n_outer_steps', 3),
                detach_outer=False  # Don't detach during eval
            )
        else:
            logits = model(inputs, n_steps=n_steps, return_all_steps=False)
        
        loss = F.cross_entropy(
            logits.permute(0, 3, 1, 2),
            targets.long()
        )
        
        preds = logits.argmax(dim=-1)
        
        # Cell-wise metrics
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_cells += targets.numel()
        
        # Puzzle-wise metrics (all cells must be correct)
        puzzles_correct = (preds == targets).view(inputs.size(0), -1).all(dim=1).sum().item()
        total_puzzles_correct += puzzles_correct
        total_puzzles += inputs.size(0)
        
        total_loss += loss.item() * inputs.size(0)
    
    avg_loss = total_loss / total_puzzles
    cell_accuracy = total_correct / total_cells
    puzzle_accuracy = total_puzzles_correct / total_puzzles
    
    metrics = {
        'loss': avg_loss,
        'cell_accuracy': cell_accuracy,
        'puzzle_accuracy': puzzle_accuracy
    }
    
    return avg_loss, metrics


def train(
    model,
    train_loader,
    val_loader,
    config
):
    """
    Main training loop.
    
    Args:
        model: The model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration dictionary
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 50),
        eta_min=config.get('lr', 1e-4) * 0.01
    )
    
    # EMA
    ema = None
    if config.get('use_ema', False):
        ema = EMA(model, decay=config.get('ema_decay', 0.9995))
        print(f"Using EMA with decay: {config.get('ema_decay', 0.9995)}")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(1, config.get('epochs', 50) + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.get('epochs', 50)}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            config,
            epoch,
            ema=ema,
            use_memory_efficient=config.get('use_memory_efficient', True)
        )
        
        scheduler.step()
        
        # Evaluate
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            config,
            use_deep_recursion=config.get('eval_use_deep_recursion', False)
        )
        
        # Evaluate with EMA if enabled
        ema_val_metrics = None
        if ema is not None:
            ema.store(model)
            ema.apply_shadow(model)
            _, ema_val_metrics = evaluate(
                model,
                val_loader,
                config,
                use_deep_recursion=config.get('eval_use_deep_recursion', False)
            )
            ema.restore(model)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Cell Acc: {val_metrics['cell_accuracy']:.4f}, "
              f"Val Puzzle Acc: {val_metrics['puzzle_accuracy']:.4f}")
        
        if ema_val_metrics is not None:
            print(f"EMA Val Cell Acc: {ema_val_metrics['cell_accuracy']:.4f}, "
                  f"EMA Val Puzzle Acc: {ema_val_metrics['puzzle_accuracy']:.4f}")
        
        # Log to wandb
        if config.get('use_wandb', False):
            log_dict = {
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'train/epoch_accuracy': train_metrics['accuracy'],
                'val/loss': val_loss,
                'val/cell_accuracy': val_metrics['cell_accuracy'],
                'val/puzzle_accuracy': val_metrics['puzzle_accuracy'],
                'lr': optimizer.param_groups[0]['lr'],
            }
            
            if ema_val_metrics is not None:
                log_dict.update({
                    'val/ema_cell_accuracy': ema_val_metrics['cell_accuracy'],
                    'val/ema_puzzle_accuracy': ema_val_metrics['puzzle_accuracy'],
                })
            
            wandb.log(log_dict)
        
        # Save best model
        current_val_acc = (ema_val_metrics if ema_val_metrics else val_metrics)['puzzle_accuracy']
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            
            # Use EMA weights if available
            if ema is not None:
                ema.store(model)
                ema.apply_shadow(model)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': ema_val_metrics if ema_val_metrics else val_metrics,
                'config': config,
            }, config.get('checkpoint_path', 'best_model.pt'))
            
            if ema is not None:
                ema.restore(model)
            
            print(f"✓ Saved best model (val_puzzle_acc: {current_val_acc:.4f})")
    
    print(f"\nTraining complete! Best val puzzle accuracy: {best_val_acc:.4f}")
    
    return best_val_acc
