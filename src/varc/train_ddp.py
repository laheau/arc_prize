import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import numpy as np
from copy import deepcopy

from .model import VARCModel
from .data import VARCDataset, collate_fn

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def train_varc_ddp(
    train_path="data/arc-agi_training_challenges.json",
    epochs=100,
    batch_size=128, # Per GPU
    lr=1e-4,
    save_path="checkpoints/varc_model_ddp.pt"
):
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"Starting DDP training on {world_size} GPUs")

    # Dataset
    dataset = VARCDataset(train_path, mode='train', augment=True)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model
    model = VARCModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs) # [B, H, W, C]
            
            loss = criterion(logits.permute(0, 3, 1, 2), targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Aggregate loss for logging
        metrics = torch.tensor([total_loss, len(dataloader)], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        avg_loss = metrics[0].item() / metrics[1].item()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                torch.save(model.module.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")
                
    cleanup_ddp()

if __name__ == "__main__":
    train_varc_ddp()
