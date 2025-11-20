import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import model and components from idea.py
from src.idea import SudokuTreeModel, EMA
from src.sudoku_extreme_pipeline import SudokuDataset, sudoku_collate_fn

def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Destroy distributed process group."""
    dist.destroy_process_group()

def train_epoch_ddp(model, dataloader, optimizer, device, epoch, ema=None, rank=0):
    model.train()
    # DDP: Set epoch for sampler shuffling
    dataloader.sampler.set_epoch(epoch)
    
    total_loss = 0.0
    total_correct = 0
    total_cells = 0
    
    # Only show progress bar on rank 0
    iterator = enumerate(dataloader)
    if rank == 0:
        iterator = tqdm(iterator, total=len(dataloader), desc=f"Epoch {epoch}")
        
    for batch_idx, batch in iterator:
        puzzles = batch["puzzle"].to(device)
        solutions = batch["solution"].to(device)
        
        optimizer.zero_grad()
        
        # Access the underlying model if wrapped in DDP
        raw_model = model.module if isinstance(model, DDP) else model
        
        B, H, W = puzzles.shape
        h = raw_model.h0.expand(B, -1, -1)
        
        batch_loss = 0.0
        logits = None
        
        # DDP Optimization: Prevent gradient sync until the final backward call
        steps = raw_model.recursion_steps
        
        # Context manager to disable gradient sync for all but the last step
        with model.no_sync():
            for i in range(steps - 1):
                embeddings = raw_model.get_embedding(puzzles)
                h, logits = raw_model.forward_step(embeddings, h)
                step_loss = F.cross_entropy(logits.permute(0, 3, 1, 2), solutions.long())
                step_loss.backward()
                batch_loss += step_loss.item()
                h = h.detach()

        # Final step (Sync gradients here)
        embeddings = raw_model.get_embedding(puzzles)
        h, logits = raw_model.forward_step(embeddings, h)
        step_loss = F.cross_entropy(logits.permute(0, 3, 1, 2), solutions.long())
        step_loss.backward()
        batch_loss += step_loss.item()
        h = h.detach()
            
        optimizer.step()
        
        if ema is not None:
            ema.update(raw_model)
        
        # Metrics
        if logits is not None:
            preds = logits.argmax(dim=-1)
            correct = (preds == solutions).sum().item()
        else:
            correct = 0
        cells = solutions.numel()
        
        total_loss += batch_loss * puzzles.size(0)
        total_correct += correct
        total_cells += cells
        
        # Logging (Rank 0 only)
        if rank == 0:
            if isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': f'{batch_loss:.4f}', 'acc': f'{correct/cells:.4f}'})
            
            if batch_idx % 50 == 0:
                wandb.log({
                    'train/loss': batch_loss,
                    'train/accuracy': correct / cells,
                    'train/step': epoch * len(dataloader) + batch_idx,
                })
    
    # Reduce metrics across all GPUs for accurate reporting
    metrics = torch.tensor([total_loss, total_correct, total_cells], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    avg_loss = metrics[0].item() / metrics[2].item() * 81 # Approx per puzzle
    avg_acc = metrics[1].item() / metrics[2].item()
    
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate_ddp(model, dataloader, device, rank=0, limit_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct_cells = 0
    total_cells = 0
    total_correct_puzzles = 0
    total_puzzles = 0
    
    iterator = dataloader
    if rank == 0:
        desc = "Evaluating (Full)" if limit_batches is None else "Evaluating (Partial)"
        iterator = tqdm(dataloader, desc=desc, total=limit_batches if limit_batches else len(dataloader))
        
    for i, batch in enumerate(iterator):
        if limit_batches is not None and i >= limit_batches:
            break
        puzzles = batch["puzzle"].to(device)
        solutions = batch["solution"].to(device)
        
        logits = model(puzzles)
        loss = F.cross_entropy(logits.permute(0, 3, 1, 2), solutions.long())
        
        preds = logits.argmax(dim=-1)
        
        total_correct_cells += (preds == solutions).sum().item()
        total_cells += solutions.numel()
        
        correct_per_puzzle = (preds == solutions).view(puzzles.size(0), -1).all(dim=1)
        total_correct_puzzles += correct_per_puzzle.sum().item()
        total_puzzles += puzzles.size(0)
        total_loss += loss.item() * puzzles.size(0)
    
    # Aggregate metrics
    metrics = torch.tensor([total_loss, total_correct_cells, total_cells, total_correct_puzzles, total_puzzles], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    avg_loss = metrics[0].item() / metrics[4].item()
    cell_acc = metrics[1].item() / metrics[2].item()
    puzzle_acc = metrics[3].item() / metrics[4].item()
    
    return avg_loss, cell_acc, puzzle_acc

def main():
    # 1. Setup DDP
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    # Hyperparameters (Should match idea.py or be loaded from config)
    config = {
        'batch_size': 128, # Per GPU batch size
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'depth': 3,
        'd_model': 256,
        'nhead': 8,
        'num_kv_heads': 2,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'use_gqa': True,
        'use_ema': True,
        'ema_decay': 0.9995,
        'recursion_steps': 16,
        'num_workers': 2, # Reduced per GPU
        'device': 'cuda',
    }
    
    # 2. Initialize WandB (Rank 0 only)
    if rank == 0:
        wandb.init(
            project="arc-prize-sudoku-tree",
            config=config,
            name=f"tree-depth{config['depth']}-d{config['d_model']}-ddp"
        )
        print(f"Training on {world_size} GPUs")
    
    # Load data
    train_dataset = SudokuDataset(split="train")
    val_dataset = SudokuDataset(split="test")
    
    # 3. Distributed Sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler, # Use sampler instead of shuffle
        collate_fn=sudoku_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        sampler=val_sampler,
        collate_fn=sudoku_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    # Create model
    model = SudokuTreeModel(
        num_classes=10,
        seq_len=81,
        d_model=config['d_model'],
        depth=config['depth'],
        nhead=config['nhead'],
        num_kv_heads=config.get('num_kv_heads'),
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        use_gqa=config.get('use_gqa', True),
        recursion_steps=config.get('recursion_steps', 1),
    ).to(device)
    
    # 4. Wrap Model with DDP
    # find_unused_parameters=False is more efficient if all params are used
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize EMA (Use model.module to access underlying model)
    ema = None
    if config.get('use_ema', False):
        ema = EMA(model.module, decay=config.get('ema_decay', 0.9995), device=device)
    
    best_val_acc = 0.0
    
    for epoch in range(1, config['epochs'] + 1):
        if rank == 0:
            print(f"\n{'='*50}\nEpoch {epoch}/{config['epochs']}\n{'='*50}")
        
        train_loss, train_acc = train_epoch_ddp(model, train_loader, optimizer, device, epoch, ema, rank)
        
        # Determine validation size
        # Full validation every 10 epochs or last epoch
        if epoch % 10 == 0 or epoch == config['epochs']:
            val_limit = None
        else:
            # Validate on approx 1000 samples
            # Global batch size = batch_size * 2 * world_size
            # 128 * 2 * 4 = 1024
            # So 1 batch per GPU is enough for ~1000 samples globally
            val_limit = 1
        
        val_loss, val_cell_acc, val_puzzle_acc = evaluate_ddp(model, val_loader, device, rank, limit_batches=val_limit)
        
        # EMA Evaluation
        ema_val_loss, ema_val_cell_acc, ema_val_puzzle_acc = None, None, None
        if ema is not None:
            ema.store(model.module)
            ema.apply_shadow(model.module)
            ema_val_loss, ema_val_cell_acc, ema_val_puzzle_acc = evaluate_ddp(model, val_loader, device, rank, limit_batches=val_limit)
            ema.restore(model.module)
        
        # Logging (Rank 0 only)
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Cell Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Cell Acc: {val_cell_acc:.4f}, Val Puzzle Acc: {val_puzzle_acc:.4f}")
            
            log_dict = {
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'train/epoch_accuracy': train_acc,
                'val/loss': val_loss,
                'val/cell_accuracy': val_cell_acc,
                'val/puzzle_accuracy': val_puzzle_acc,
            }
            
            if ema is not None:
                log_dict.update({
                    'val/ema_loss': ema_val_loss,
                    'val/ema_cell_accuracy': ema_val_cell_acc,
                    'val/ema_puzzle_accuracy': ema_val_puzzle_acc,
                })
            
            wandb.log(log_dict)
            
            # Save checkpoint
            current_val_acc = ema_val_puzzle_acc if ema is not None and ema_val_puzzle_acc is not None else val_puzzle_acc
            if current_val_acc is not None and current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                if ema is not None:
                    ema.store(model.module)
                    ema.apply_shadow(model.module)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), # Save module state dict
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }, 'checkpoints/best_tree_model_ddp.pt')
                
                if ema is not None:
                    ema.restore(model.module)
                print(f"✓ Saved best model")

    cleanup_ddp()

if __name__ == "__main__":
    main()
