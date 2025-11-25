import torch
import numpy as np
from src.varc.model import VARCModel
from src.varc.train import test_time_training, predict_task

def smoke_test():
    print("Running smoke test on CPU...")
    device = 'cpu'
    
    # 1. Initialize Model
    model = VARCModel(
        canvas_size=32,
        num_colors=12,
        d_model=64,      # Small model for test
        nhead=4,
        num_layers=2,
        dim_feedforward=128
    ).to(device)
    print("Model initialized.")
    
    # 2. Create Dummy Task
    # A simple task: Identity (Input == Output)
    task = {
        'train': [
            {'input': [[1, 0], [0, 1]], 'output': [[1, 0], [0, 1]]},
            {'input': [[2, 2], [2, 2]], 'output': [[2, 2], [2, 2]]}
        ],
        'test': [
            {'input': [[3, 0], [0, 3]]} # Expected output: [[3, 0], [0, 3]]
        ]
    }
    print("Dummy task created.")
    
    # 3. Test Forward Pass
    dummy_input = torch.randint(0, 10, (1, 32, 32)).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Forward pass successful. Output shape: {output.shape}")
    
    # 4. Test TTT
    print("Testing TTT...")
    model_adapted = test_time_training(model, task, steps=5, lr=1e-3, device=device)
    print("TTT successful.")
    
    # 5. Test Prediction
    print("Testing Prediction...")
    preds = predict_task(model_adapted, task, device=device)
    print(f"Prediction successful. Result: {preds}")
    
    print("Smoke test passed!")

if __name__ == "__main__":
    smoke_test()
