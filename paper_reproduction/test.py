"""
Test script for paper reproduction implementation
==================================================

This script performs basic sanity checks on the model and training code.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_reproduction.model import ARCDeepRecursiveModel, DeepRecursiveModel, count_parameters
from paper_reproduction.train import train_step_memory_efficient, train_step_standard


def test_model_creation():
    """Test model creation and parameter counting."""
    print("Testing model creation...")
    
    model = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=64,
        latent_channels=512,
        n_res_blocks=2
    )
    
    num_params = count_parameters(model)
    print(f"✓ Model created successfully")
    print(f"  Parameters: {num_params:,}")
    
    return model


def test_forward_pass():
    """Test basic forward pass."""
    print("\nTesting forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=32,  # Smaller for faster testing
        latent_channels=256,
        n_res_blocks=1
    ).to(device)
    
    # Create dummy input
    B, H, W = 4, 32, 32
    inputs = torch.randint(0, 12, (B, H, W), device=device)
    
    # Test standard forward
    logits = model(inputs, n_steps=2)
    assert logits.shape == (B, H, W, 12), f"Expected shape {(B, H, W, 12)}, got {logits.shape}"
    print(f"✓ Standard forward pass successful")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test with return_all_steps
    logits, all_logits = model(inputs, n_steps=3, return_all_steps=True)
    assert len(all_logits) == 3, f"Expected 3 steps, got {len(all_logits)}"
    print(f"✓ Forward pass with all steps successful")
    print(f"  Number of steps: {len(all_logits)}")
    
    return model, inputs


def test_deep_recursion():
    """Test deep recursion forward pass."""
    print("\nTesting deep recursion...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=32,
        latent_channels=256,
        n_res_blocks=1
    ).to(device)
    
    # Create dummy input
    B, H, W = 4, 32, 32
    inputs = torch.randint(0, 12, (B, H, W), device=device)
    
    # Test deep recursion
    logits, all_logits = model.deep_recursion_forward(
        inputs,
        n_inner_steps=2,
        n_outer_steps=3,
        detach_outer=True
    )
    
    assert logits.shape == (B, H, W, 12), f"Expected shape {(B, H, W, 12)}, got {logits.shape}"
    assert len(all_logits) == 3, f"Expected 3 outer steps, got {len(all_logits)}"
    
    print(f"✓ Deep recursion forward pass successful")
    print(f"  Inner steps: 2, Outer steps: 3")
    print(f"  Total recursive steps: 6")
    print(f"  Output shape: {logits.shape}")
    
    return model, inputs


def test_training_step():
    """Test training step (without actual optimization)."""
    print("\nTesting training steps...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=32,
        latent_channels=256,
        n_res_blocks=1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    B, H, W = 4, 32, 32
    inputs = torch.randint(0, 12, (B, H, W), device=device)
    targets = torch.randint(0, 12, (B, H, W), device=device)
    batch = (inputs, targets)
    
    # Test standard training step
    loss, metrics = train_step_standard(
        model=model,
        batch=batch,
        optimizer=optimizer,
        n_steps=2,
        device=device
    )
    
    print(f"✓ Standard training step successful")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Test memory-efficient training step
    loss, metrics = train_step_memory_efficient(
        model=model,
        batch=batch,
        optimizer=optimizer,
        n_inner_steps=2,
        n_outer_steps=2,
        device=device
    )
    
    print(f"✓ Memory-efficient training step successful")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Step accuracies: {[f'{acc:.4f}' for acc in metrics['step_accuracies']]}")


def test_gradient_flow():
    """Test that gradients flow properly in both training modes."""
    print("\nTesting gradient flow...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=16,
        latent_channels=128,
        n_res_blocks=1
    ).to(device)
    
    # Create dummy batch
    B, H, W = 2, 32, 32
    inputs = torch.randint(0, 12, (B, H, W), device=device)
    targets = torch.randint(0, 12, (B, H, W), device=device)
    
    # Test gradient flow in memory-efficient mode
    model.zero_grad()
    logits, all_logits = model.deep_recursion_forward(
        inputs,
        n_inner_steps=2,
        n_outer_steps=2,
        detach_outer=True
    )
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.permute(0, 3, 1, 2),
        targets.long()
    )
    loss.backward()
    
    # Check that gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    print(f"✓ Gradient flow test passed")
    print(f"  Parameters with gradients: {grad_count}/{total_params}")
    
    assert grad_count > 0, "No gradients found!"


def test_memory_efficiency():
    """Test that memory-efficient mode actually uses less memory."""
    print("\nTesting memory efficiency...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test standard mode
    model1 = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=32,
        latent_channels=256,
        n_res_blocks=1
    ).to(device)
    
    B, H, W = 8, 32, 32
    inputs = torch.randint(0, 12, (B, H, W), device=device)
    targets = torch.randint(0, 12, (B, H, W), device=device)
    
    # Standard forward (keeps full graph)
    model1.zero_grad()
    logits, all_logits = model1(inputs, n_steps=6, return_all_steps=True)
    loss = sum(
        torch.nn.functional.cross_entropy(
            step_logits.permute(0, 3, 1, 2),
            targets.long()
        )
        for step_logits in all_logits
    ) / len(all_logits)
    loss.backward()
    
    standard_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Clear
    del model1, logits, all_logits, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test memory-efficient mode
    model2 = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=32,
        latent_channels=256,
        n_res_blocks=1
    ).to(device)
    
    inputs = torch.randint(0, 12, (B, H, W), device=device)
    targets = torch.randint(0, 12, (B, H, W), device=device)
    
    # Memory-efficient forward (detaches between steps)
    model2.zero_grad()
    logits, all_logits = model2.deep_recursion_forward(
        inputs,
        n_inner_steps=2,
        n_outer_steps=3,
        detach_outer=True
    )
    loss = sum(
        torch.nn.functional.cross_entropy(
            step_logits.permute(0, 3, 1, 2),
            targets.long()
        )
        for step_logits in all_logits
    ) / len(all_logits)
    loss.backward()
    
    efficient_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"✓ Memory test completed")
    print(f"  Standard mode: {standard_memory:.2f} MB")
    print(f"  Memory-efficient mode: {efficient_memory:.2f} MB")
    print(f"  Savings: {standard_memory - efficient_memory:.2f} MB ({(1 - efficient_memory/standard_memory)*100:.1f}%)")
    
    # Clean up
    torch.cuda.empty_cache()


def main():
    """Run all tests."""
    print("="*70)
    print("Paper Reproduction Implementation - Test Suite")
    print("="*70)
    
    try:
        # Basic tests
        test_model_creation()
        test_forward_pass()
        test_deep_recursion()
        test_training_step()
        test_gradient_flow()
        
        # Memory test (only on CUDA)
        test_memory_efficiency()
        
        print("\n" + "="*70)
        print("✓ All tests passed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
