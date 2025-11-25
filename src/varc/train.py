import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
import json
from .model import VARCModel
from .data import VARCDataset, collate_fn, pad_grid

def train_varc(
    train_path="data/arc-agi_training_challenges.json",
    epochs=100,
    batch_size=128,
    lr=1e-4,
    device='cuda',
    save_path="checkpoints/varc_model.pt"
):
    dataset = VARCDataset(train_path, mode='train', augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    model = VARCModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs) # [B, H, W, C]
            
            loss = criterion(logits.permute(0, 3, 1, 2), targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

def test_time_training(
    model,
    task,
    steps=50,
    lr=1e-4,
    device='cuda'
):
    """
    Perform TTT on a single task.
    task: dict with 'train' and 'test' examples.
    """
    # Clone model to avoid modifying the base model
    model_ttt = deepcopy(model)
    model_ttt.train()
    
    # Setup optimizer for TTT
    optimizer = optim.AdamW(model_ttt.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare support set (train examples of the task)
    train_examples = task['train']
    inputs = []
    outputs = []
    
    # We should augment the support set during TTT as well to prevent overfitting
    # Or just iterate multiple times.
    # The paper says "generalizes to unseen tasks through test-time training".
    # Usually this means overfitting to the support set.
    
    # Let's do simple optimization on support set
    # We need to handle variable sizes in support set by padding
    
    for _ in range(steps):
        # Sample augmentations for support set
        batch_in = []
        batch_out = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Augment
            k = np.random.randint(0, 8)
            # Simple dihedral for TTT
            if k == 1: inp, out = np.rot90(inp), np.rot90(out)
            elif k == 2: inp, out = np.rot90(inp, 2), np.rot90(out, 2)
            elif k == 3: inp, out = np.rot90(inp, 3), np.rot90(out, 3)
            elif k == 4: inp, out = np.fliplr(inp), np.fliplr(out)
            elif k == 5: inp, out = np.flipud(inp), np.flipud(out)
            elif k == 6: inp, out = inp.T, out.T
            
            # Pad
            inp_pad, _, _ = pad_grid(inp, 32)
            out_pad, _, _ = pad_grid(out, 32)
            
            batch_in.append(inp_pad)
            batch_out.append(out_pad)
            
        t_in = torch.LongTensor(np.stack(batch_in)).to(device)
        t_out = torch.LongTensor(np.stack(batch_out)).to(device)
        
        optimizer.zero_grad()
        logits = model_ttt(t_in)
        loss = criterion(logits.permute(0, 3, 1, 2), t_out)
        loss.backward()
        optimizer.step()
        
    return model_ttt

def predict_task(model, task, device='cuda'):
    """
    Predict test examples for a task using the model.
    """
    model.eval()
    predictions = []
    
    for ex in task['test']:
        inp = np.array(ex['input'])
        # Pad
        inp_pad, h, w = pad_grid(inp, 32)
        t_in = torch.LongTensor(inp_pad).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(t_in) # [1, 32, 32, C]
            pred_grid = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            
        # Crop back
        pred_grid = pred_grid[:h, :w]
        predictions.append(pred_grid.tolist())
        
    return predictions

def evaluate_varc(
    model_path,
    eval_path="data/arc-agi_evaluation_challenges.json",
    sol_path="data/arc-agi_evaluation_solutions.json",
    device='cuda',
    ttt_steps=50
):
    # Load model
    model = VARCModel().to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Load data
    eval_data = json.load(open(eval_path))
    sol_data = json.load(open(sol_path))
    
    total = 0
    correct = 0
    
    print(f"Evaluating on {len(eval_data)} tasks with TTT steps={ttt_steps}...")
    
    for tid, task in eval_data.items():
        # Perform TTT
        model_adapted = test_time_training(model, task, steps=ttt_steps, device=device)
        
        # Predict
        preds = predict_task(model_adapted, task, device=device)
        
        # Check correctness
        # ARC evaluation usually allows 3 guesses, here we just output 1 for simplicity
        # Or we could implement voting with augmentations.
        
        solutions = sol_data[tid]
        task_correct = True
        
        for i, pred in enumerate(preds):
            sol = solutions[i]
            if pred != sol:
                task_correct = False
                break
        
        if task_correct:
            correct += 1
        total += 1
        
        print(f"Task {tid}: {'Correct' if task_correct else 'Wrong'} ({correct}/{total})")
        
    print(f"Final Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    # Example usage
    # train_varc()
    pass
