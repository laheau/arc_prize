import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import wandb
from tqdm import tqdm
import math
from copy import deepcopy


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) - a middle ground between MHA and MQA.
    
    In GQA, queries have nhead heads, but keys and values are shared across groups.
    num_kv_heads determines the number of key/value heads (nhead must be divisible by num_kv_heads).
    
    When num_kv_heads = nhead: Standard Multi-Head Attention
    When num_kv_heads = 1: Multi-Query Attention (MQA)
    When 1 < num_kv_heads < nhead: Grouped Query Attention (GQA)
    """
    
    def __init__(self, d_model, nhead, num_kv_heads=None, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else nhead
        assert nhead % self.num_kv_heads == 0, "nhead must be divisible by num_kv_heads"
        
        self.head_dim = d_model // nhead
        self.num_groups = nhead // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        B, L, D = x.shape
        
        # Project and reshape
        Q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, L, head_dim]
        K = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
        V = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
        
        # Expand K, V to match Q's head count by repeating across groups
        K = K.repeat_interleave(self.num_groups, dim=1)  # [B, nhead, L, head_dim]
        V = V.repeat_interleave(self.num_groups, dim=1)  # [B, nhead, L, head_dim]
        
        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale  # [B, nhead, L, L]
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights @ V  # [B, nhead, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_proj(out)
        
        return out


class GQATransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer using Grouped Query Attention instead of standard MHA."""
    
    def __init__(self, d_model, nhead, num_kv_heads=None, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Self-attention with GQA
        self.self_attn = GroupedQueryAttention(d_model, nhead, num_kv_heads, dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.self_attn(src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class InternalTreeNode(nn.Module):
    def __init__(self, rec_count, child, seq_len, d_model, dim_feedforward=512, dropout=0.1):
        super(InternalTreeNode, self).__init__()
        self.rec_count = rec_count

        self.out_proj = nn.Linear(d_model, d_model)
        
        # MLP to generate initial y from x
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.child = child

    def forward(self, x):
        y = self.mlp(x)
        for _ in range(self.rec_count):
            y = self.child(self.norm(x + y))
        return self.out_proj(self.norm(x + y))


class Tree(nn.Module):
    def __init__(self, depth, seq_len=81, d_model=128, nhead=8, num_kv_heads=None, dim_feedforward=512, dropout=0.1, use_gqa=True):
        super(Tree, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_gqa = use_gqa
        
        if use_gqa:
            # Use custom GQA-based transformer
            encoder_layers = [
                GQATransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    num_kv_heads=num_kv_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
                for _ in range(2)
            ]
            self.leaf = nn.Sequential(*encoder_layers)
        else:
            # Use standard PyTorch transformer
            self.leaf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=2
            )
        
        node = self.leaf
        for _ in range(depth):
            parent = InternalTreeNode(
                rec_count=2, 
                child=node, 
                seq_len=seq_len, 
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            node = parent
        self.root = node

    def forward(self, x):
        return self.root(x)


class SudokuTreeModel(nn.Module):
    """Full model: embedding -> tree -> prediction head."""
    
    def __init__(self, num_classes=10, seq_len=81, d_model=128, depth=5, nhead=8, num_kv_heads=None, 
                 dim_feedforward=512, dropout=0.1, use_gqa=True, recursion_steps=1):
        super(SudokuTreeModel, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.recursion_steps = recursion_steps
        
        # Embedding layer: maps 0-9 to d_model
        self.embedding = nn.Embedding(num_classes, d_model)
        
        # 2D Positional encoding for 9x9 grid
        self.pos_encoding = self._create_2d_positional_encoding(9, 9, d_model)
        
        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.h0, std=0.02)
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Tree processor
        self.tree = Tree(
            depth=depth, 
            seq_len=seq_len, 
            d_model=d_model, 
            nhead=nhead,
            num_kv_heads=num_kv_heads,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            use_gqa=use_gqa
        )
        
        # Output head: d_model -> num_classes
        self.head = nn.Linear(d_model, num_classes)
    
    def _create_2d_positional_encoding(self, height, width, d_model):
        """Create 2D sinusoidal positional encodings for a grid."""
        pe = torch.zeros(height * width, d_model)
        
        # Create position indices
        position_h = torch.arange(height).unsqueeze(1).repeat(1, width).flatten().unsqueeze(1)  # [H*W, 1]
        position_w = torch.arange(width).unsqueeze(0).repeat(height, 1).flatten().unsqueeze(1)  # [H*W, 1]
        
        # Create frequency indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sin/cos to different dimensions
        # First half of features: encode height
        pe[:, 0:d_model//2:2] = torch.sin(position_h * div_term[:d_model//4])
        pe[:, 1:d_model//2:2] = torch.cos(position_h * div_term[:d_model//4])
        
        # Second half of features: encode width
        pe[:, d_model//2::2] = torch.sin(position_w * div_term[:d_model//4])
        pe[:, d_model//2+1::2] = torch.cos(position_w * div_term[:d_model//4])
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # [1, seq_len, d_model]
    
    def get_embedding(self, x):
        """Get initial embeddings from input puzzle."""
        B, H, W = x.shape
        x_flat = x.view(B, H * W)  # [B, seq_len]
        
        # Embed
        embeddings = self.embedding(x_flat)  # [B, seq_len, d_model]
        
        # Add positional encoding
        embeddings = embeddings + self.pos_encoding.to(embeddings.device)  # [B, seq_len, d_model]
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        return embeddings

    def forward_step(self, embeddings, h):
        """Perform one step of the recursive model."""
        # Input is sum of task embedding (x) and previous output (y/h)
        input_feat = embeddings + h
        
        # Forward pass through tree
        h_new = self.tree(input_feat)
        
        # Compute logits
        logits = self.head(h_new)
        B = embeddings.shape[0]
        logits = logits.view(B, 9, 9, -1)
        
        return h_new, logits

    def forward(self, x, return_all_steps=False):
        """
        Args:
            x: [B, H, W] long tensor with values 0-9
            return_all_steps: If True, returns a list of logits for each recursion step.
        Returns:
            logits: [B, H, W, num_classes] or list of logits
        """
        embeddings = self.get_embedding(x)
        h = self.h0.expand(embeddings.size(0), -1, -1)
        outputs = []
        
        # Determine number of steps
        steps = self.recursion_steps
        
        for _ in range(steps):
            h, logits = self.forward_step(embeddings, h)
            outputs.append(logits)
            
        if return_all_steps:
            return outputs
            
        return outputs[-1]


class EMA:
    """Exponential Moving Average of model parameters.
    
    Args:
        model: The model to track
        decay: The decay rate (default: 0.999)
        device: Device to store the shadow parameters
    """
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.device = device
        # Create a copy of the model for the shadow parameters
        self.shadow = deepcopy(model).eval()
        self.shadow.to(device)
        
        # Disable gradients for shadow model
        for param in self.shadow.parameters():
            param.requires_grad = False
        
        self.num_updates = 0
    
    @torch.no_grad()
    def update(self, model):
        """Update the moving average with the current model parameters."""
        self.num_updates += 1
        # Use a dynamic decay that starts lower and increases
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            if model_param.requires_grad:
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def apply_shadow(self, model):
        """Copy EMA parameters to model."""
        for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            model_param.data.copy_(ema_param.data)
    
    def store(self, model):
        """Save current model parameters (before applying shadow)."""
        self.backup = {}
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()
    
    def restore(self, model):
        """Restore model parameters from backup."""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


def train_epoch(model, dataloader, optimizer, device, epoch, ema=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_cells = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=True)
    for batch_idx, batch in enumerate(pbar):
        puzzles = batch["puzzle"].to(device)  # [B, 9, 9]
        solutions = batch["solution"].to(device)  # [B, 9, 9]
        
        optimizer.zero_grad()
        
        # Manual recursion loop for memory efficiency
        # We compute gradients at each step and accumulate them, freeing the graph after each step
        
        # Initialize h
        B, H, W = puzzles.shape
        h = model.h0.expand(B, -1, -1)
        
        batch_loss = 0.0
        logits = None
        
        for _ in range(model.recursion_steps):
            # Recompute embeddings to allow backprop through them at each step without retain_graph=True
            embeddings = model.get_embedding(puzzles)
            
            h, logits = model.forward_step(embeddings, h)
            
            step_loss = F.cross_entropy(
                logits.permute(0, 3, 1, 2),  # [B, C, H, W]
                solutions.long()
            )
            
            # Backward pass for this step
            # This accumulates gradients and frees the graph for this step
            step_loss.backward()
            batch_loss += step_loss.item()
            
            # Detach h for the next step to prevent backprop through time (TBPTT k=1)
            # This is necessary because the graph for the previous step is freed
            h = h.detach()
            
        optimizer.step()
        
        # Update EMA after optimizer step
        if ema is not None:
            ema.update(model)
        
        # Metrics (use final prediction)
        if logits is not None:
            preds = logits.argmax(dim=-1)
            correct = (preds == solutions).sum().item()
        else:
            correct = 0
        cells = solutions.numel()
        
        total_loss += batch_loss * puzzles.size(0)
        total_correct += correct
        total_cells += cells
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{correct/cells:.4f}'
        })
        
        # Log to wandb every 50 steps
        if batch_idx % 50 == 0:
            wandb.log({
                'train/loss': batch_loss,
                'train/accuracy': correct / cells,
                'train/step': epoch * len(dataloader) + batch_idx,
            })
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / total_cells
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, device, limit_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct_cells = 0
    total_cells = 0
    total_correct_puzzles = 0
    total_puzzles = 0
    
    desc = "Evaluating (Full)" if limit_batches is None else "Evaluating (Partial)"
    iterator = tqdm(dataloader, desc=desc, total=limit_batches if limit_batches else len(dataloader), disable=True)
    
    for i, batch in enumerate(iterator):
        if limit_batches is not None and i >= limit_batches:
            break
        puzzles = batch["puzzle"].to(device)
        solutions = batch["solution"].to(device)
        
        logits = model(puzzles)
        loss = F.cross_entropy(
            logits.permute(0, 3, 1, 2),
            solutions.long()
        )
        
        preds = logits.argmax(dim=-1)
        
        # Cell-wise accuracy
        total_correct_cells += (preds == solutions).sum().item()
        total_cells += solutions.numel()
        
        # Puzzle-wise accuracy (all cells must be correct)
        correct_per_puzzle = (preds == solutions).view(puzzles.size(0), -1).all(dim=1)
        total_correct_puzzles += correct_per_puzzle.sum().item()
        total_puzzles += puzzles.size(0)
        
        total_loss += loss.item() * puzzles.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    cell_acc = total_correct_cells / total_cells
    puzzle_acc = total_correct_puzzles / total_puzzles
    
    return avg_loss, cell_acc, puzzle_acc


def main():
    # Hyperparameters
    config = {
        'batch_size': 128,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'depth': 3,
        'd_model': 128,
        'nhead': 16,
        'num_kv_heads': 2,  # GQA: 16 query heads, 2 key/value heads
        'dim_feedforward': 512,
        'dropout': 0.1,
        'use_gqa': True,  # Set to False to use standard multi-head attention
        'use_ema': True,  # Use Exponential Moving Average
        'ema_decay': 0.9995,  # EMA decay rate
        'recursion_steps': 3, # Number of recursive steps (1 = standard)
        'num_workers': 6,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Initialize wandb
    wandb.init(
        project="arc-prize-sudoku-tree",
        config=config,
        name=f"tree-depth{config['depth']}-d{config['d_model']}"
    )
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Load data
    from sudoku_extreme_pipeline import SudokuDataset, sudoku_collate_fn
    
    print("Loading datasets...")
    train_dataset = SudokuDataset(split="train")
    # train_dataset = Subset(train_dataset, range(100))  # Limit to 100 samples for testing
    
    # Use a small subset of test for validation (test set is smaller)
    val_dataset = SudokuDataset(split="test")
    # val_dataset = Subset(val_dataset, range(50))  # Limit to 50 samples for testing
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=sudoku_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        collate_fn=sudoku_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    wandb.watch(model, log='all', log_freq=100)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize EMA
    ema = None
    if config.get('use_ema', False):
        ema = EMA(model, decay=config.get('ema_decay', 0.9995), device=device)
        print(f"Using EMA with decay: {config.get('ema_decay', 0.9995)}")
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*50}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch, ema)
        
        # Determine validation size
        if epoch % 10 == 0 or epoch == config['epochs']:
            val_limit = None
        else:
            val_limit = 1000 // (config['batch_size'] * 2)
            if val_limit < 1: val_limit = 1
            
        # Evaluate with regular model
        val_loss, val_cell_acc, val_puzzle_acc = evaluate(model, val_loader, device, limit_batches=val_limit)
        
        # Evaluate with EMA model if enabled
        ema_val_loss, ema_val_cell_acc, ema_val_puzzle_acc = None, None, None
        if ema is not None:
            # Temporarily swap to EMA weights for evaluation
            ema.store(model)
            ema.apply_shadow(model)
            ema_val_loss, ema_val_cell_acc, ema_val_puzzle_acc = evaluate(model, val_loader, device, limit_batches=val_limit)
            ema.restore(model)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Cell Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Cell Acc: {val_cell_acc:.4f}, Val Puzzle Acc: {val_puzzle_acc:.4f}")
        
        if ema is not None:
            print(f"EMA Val Loss: {ema_val_loss:.4f}, EMA Val Cell Acc: {ema_val_cell_acc:.4f}, EMA Val Puzzle Acc: {ema_val_puzzle_acc:.4f}")
        
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
        
        # Save best model (based on puzzle accuracy)
        # Use EMA accuracy if available, otherwise use regular accuracy
        current_val_acc = ema_val_puzzle_acc if ema is not None and ema_val_puzzle_acc is not None else val_puzzle_acc
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            
            # Save with EMA weights if using EMA
            if ema is not None:
                ema.store(model)
                ema.apply_shadow(model)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cell_acc': ema_val_cell_acc if ema is not None else val_cell_acc,
                'val_puzzle_acc': ema_val_puzzle_acc if ema is not None else val_puzzle_acc,
                'config': config,
            }, 'checkpoints/best_tree_model.pt')
            
            if ema is not None:
                ema.restore(model)
            
            print(f"✓ Saved best model (val_puzzle_acc: {current_val_acc:.4f})")
    
    wandb.finish()
    print(f"\nTraining complete! Best val puzzle accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

