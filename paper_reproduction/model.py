"""
Deep Recursion Model with Gradient Detachment for ARC Prize
============================================================

This implementation reproduces key techniques from the paper:
1. Deep recursion with gradient detachment between steps
2. Task + output vector summing for model inputs
3. Trajectory rollout for multiple gradient updates
4. Memory-efficient training with intermediate gradient steps

Based on the "Less is More" recursive model approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.dropout(out)
        return F.gelu(out + residual)


class Encoder(nn.Module):
    """Encoder network that processes input grids to latent representations."""
    
    def __init__(self, in_channels=12, base_channels=64, latent_channels=512, n_res_blocks=2):
        super().__init__()
        
        self.stem = ConvBlock(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            *[ResidualBlock(base_channels * 2) for _ in range(n_res_blocks)]
        )
        
        self.down2 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            *[ResidualBlock(base_channels * 4) for _ in range(n_res_blocks)]
        )
        
        self.down3 = nn.Sequential(
            ConvBlock(base_channels * 4, latent_channels, kernel_size=3, stride=2, padding=1),
            *[ResidualBlock(latent_channels) for _ in range(n_res_blocks)]
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(latent_channels) for _ in range(n_res_blocks)]
        )
        
    def forward(self, x):
        # x: [B, in_channels, H, W]
        x0 = self.stem(x)           # [B, base, H, W]
        x1 = self.down1(x0)         # [B, base*2, H/2, W/2]
        x2 = self.down2(x1)         # [B, base*4, H/4, W/4]
        x3 = self.down3(x2)         # [B, latent, H/8, W/8]
        z = self.bottleneck(x3)     # [B, latent, H/8, W/8]
        
        return z, (x0, x1, x2)


class Decoder(nn.Module):
    """Decoder network with skip connections (U-Net style)."""
    
    def __init__(self, out_channels=12, base_channels=64, latent_channels=512, n_res_blocks=2):
        super().__init__()
        
        # Upsampling path with skip connections
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            *[ResidualBlock(base_channels * 4) for _ in range(n_res_blocks)]
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(base_channels * 8, base_channels * 2, kernel_size=3, padding=1),
            *[ResidualBlock(base_channels * 2) for _ in range(n_res_blocks)]
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(base_channels * 4, base_channels, kernel_size=3, padding=1),
            *[ResidualBlock(base_channels) for _ in range(n_res_blocks)]
        )
        
        # Output head
        self.head = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, z, skips):
        # z: [B, latent, H/8, W/8]
        # skips: (x0, x1, x2)
        x0, x1, x2 = skips
        
        x = self.up3(z)                      # [B, base*4, H/4, W/4]
        x = torch.cat([x, x2], dim=1)        # [B, base*8, H/4, W/4]
        
        x = self.up2(x)                      # [B, base*2, H/2, W/2]
        x = torch.cat([x, x1], dim=1)        # [B, base*4, H/2, W/2]
        
        x = self.up1(x)                      # [B, base, H, W]
        x = torch.cat([x, x0], dim=1)        # [B, base*2, H, W]
        
        out = self.head(x)                   # [B, out_channels, H, W]
        return out


class DeepRecursiveModel(nn.Module):
    """
    Deep Recursive Model for ARC tasks.
    
    Key features:
    1. Takes task input (x) and current output (y) as input
    2. Processes through encoder-decoder architecture
    3. Supports recursive refinement with gradient detachment
    4. Memory-efficient training with intermediate gradient steps
    
    Args:
        in_channels: Number of input channels (default: 12 for one-hot encoded colors)
        out_channels: Number of output channels (default: 12 for color predictions)
        base_channels: Base number of channels for the network
        latent_channels: Number of channels in the latent representation
        n_res_blocks: Number of residual blocks per stage
    """
    
    def __init__(
        self,
        in_channels=12,
        out_channels=12,
        base_channels=64,
        latent_channels=512,
        n_res_blocks=2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        
        # Task and output embeddings
        self.task_proj = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        
        # Main encoder-decoder
        self.encoder = Encoder(
            in_channels=base_channels * 2,  # task + output
            base_channels=base_channels,
            latent_channels=latent_channels,
            n_res_blocks=n_res_blocks
        )
        
        self.decoder = Decoder(
            out_channels=out_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            n_res_blocks=n_res_blocks
        )
        
        # Learnable initial output state
        self.y_init = nn.Parameter(torch.zeros(1, in_channels, 32, 32))
        nn.init.normal_(self.y_init, std=0.02)
        
        # Latent state projection
        self.z_proj = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        
    def forward_step(self, task_input, current_output, previous_z=None):
        """
        Single forward step of the recursive model.
        
        Args:
            task_input: Task input grid [B, C, H, W]
            current_output: Current output state [B, C, H, W]
            previous_z: Previous latent state (optional) [B, latent_C, H/8, W/8]
            
        Returns:
            new_output: Updated output logits [B, C, H, W]
            new_z: New latent state [B, latent_C, H/8, W/8]
        """
        # Project and sum task and output (key technique from the paper)
        task_feat = self.task_proj(task_input)
        output_feat = self.output_proj(current_output)
        
        # Concatenate task and output features
        combined = torch.cat([task_feat, output_feat], dim=1)
        
        # Encode to latent space
        z, skips = self.encoder(combined)
        
        # Add previous latent state if available (residual connection in latent space)
        if previous_z is not None:
            z = z + self.z_proj(previous_z)
        
        # Decode to output
        output = self.decoder(z, skips)
        
        return output, z
    
    def latent_recursion(self, task_input, current_output, z, n_steps=3):
        """
        Perform n steps of latent recursion.
        
        Args:
            task_input: Task input grid [B, C, H, W]
            current_output: Current output state [B, C, H, W]
            z: Current latent state [B, latent_C, H/8, W/8]
            n_steps: Number of recursion steps
            
        Returns:
            final_output: Final output after n steps [B, C, H, W]
            final_z: Final latent state [B, latent_C, H/8, W/8]
        """
        for _ in range(n_steps):
            current_output, z = self.forward_step(task_input, current_output, z)
        
        return current_output, z
    
    def deep_recursion(
        self,
        task_input,
        n_inner_steps=3,
        n_outer_steps=3,
        detach_outer=True
    ):
        """
        Deep recursion with gradient detachment (key technique from paper).
        
        Performs multiple outer steps with gradient detachment between them,
        and multiple inner steps within each outer step.
        
        Args:
            task_input: Task input grid [B, C, H, W]
            n_inner_steps: Number of inner recursion steps per outer step
            n_outer_steps: Number of outer steps (with gradient detachment)
            detach_outer: Whether to detach gradients between outer steps
            
        Returns:
            output: Final output logits [B, C, H, W]
            all_outputs: List of outputs from each outer step (for training)
        """
        B = task_input.size(0)
        device = task_input.device
        
        # Initialize output with learnable initial state
        current_output = self.y_init.expand(B, -1, -1, -1).to(device)
        z = None
        
        all_outputs = []
        
        for i in range(n_outer_steps):
            # Perform inner recursion steps
            current_output, z = self.latent_recursion(
                task_input, current_output, z, n_steps=n_inner_steps
            )
            
            all_outputs.append(current_output)
            
            # Detach gradients between outer steps (except for the last step)
            # This is the key memory-efficient technique from the paper
            if detach_outer and i < n_outer_steps - 1:
                current_output = current_output.detach()
                z = z.detach()
        
        return current_output, all_outputs
    
    def forward(self, task_input, n_steps=1, return_all_steps=False):
        """
        Forward pass with configurable number of recursion steps.
        
        Args:
            task_input: Task input grid [B, C, H, W]
            n_steps: Number of recursion steps (default: 1)
            return_all_steps: Whether to return outputs from all steps
            
        Returns:
            output: Final output logits [B, C, H, W]
            (optional) all_outputs: List of outputs from each step
        """
        B = task_input.size(0)
        device = task_input.device
        
        current_output = self.y_init.expand(B, -1, -1, -1).to(device)
        z = None
        all_outputs = []
        
        for _ in range(n_steps):
            current_output, z = self.forward_step(task_input, current_output, z)
            all_outputs.append(current_output)
        
        if return_all_steps:
            return current_output, all_outputs
        return current_output


class ARCDeepRecursiveModel(nn.Module):
    """
    Wrapper model for ARC tasks that handles input preprocessing.
    
    This model:
    1. Converts discrete color grids to one-hot representations
    2. Applies the deep recursive model
    3. Returns class predictions
    """
    
    def __init__(
        self,
        num_colors=10,
        base_channels=64,
        latent_channels=512,
        n_res_blocks=2
    ):
        super().__init__()
        
        self.num_colors = num_colors
        # +2 for padding indicator and background
        self.in_channels = num_colors + 2
        
        self.model = DeepRecursiveModel(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            n_res_blocks=n_res_blocks
        )
        
    def to_onehot(self, x):
        """Convert discrete grid to one-hot representation."""
        B, H, W = x.shape
        # One-hot encode
        x_onehot = F.one_hot(x.long(), num_classes=self.in_channels)  # [B, H, W, C]
        x_onehot = x_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        return x_onehot
    
    def forward(self, task_input, n_steps=1, return_all_steps=False):
        """
        Forward pass for ARC tasks.
        
        Args:
            task_input: Discrete grid [B, H, W] with values in [0, num_colors+1]
            n_steps: Number of recursion steps
            return_all_steps: Whether to return outputs from all steps
            
        Returns:
            logits: Class predictions [B, H, W, C]
            (optional) all_logits: List of predictions from each step
        """
        # Convert to one-hot
        task_onehot = self.to_onehot(task_input)
        
        # Forward through model
        if return_all_steps:
            output, all_outputs = self.model(task_onehot, n_steps, return_all_steps=True)
            # Convert all outputs to class predictions
            all_logits = [out.permute(0, 2, 3, 1) for out in all_outputs]  # [B, H, W, C]
            return output.permute(0, 2, 3, 1), all_logits
        else:
            output = self.model(task_onehot, n_steps, return_all_steps=False)
            return output.permute(0, 2, 3, 1)  # [B, H, W, C]
    
    def deep_recursion_forward(
        self,
        task_input,
        n_inner_steps=3,
        n_outer_steps=3,
        detach_outer=True
    ):
        """
        Deep recursion forward pass for ARC tasks.
        
        Args:
            task_input: Discrete grid [B, H, W]
            n_inner_steps: Number of inner recursion steps
            n_outer_steps: Number of outer steps with gradient detachment
            detach_outer: Whether to detach gradients between outer steps
            
        Returns:
            logits: Final class predictions [B, H, W, C]
            all_logits: List of predictions from each outer step
        """
        task_onehot = self.to_onehot(task_input)
        
        output, all_outputs = self.model.deep_recursion(
            task_onehot,
            n_inner_steps=n_inner_steps,
            n_outer_steps=n_outer_steps,
            detach_outer=detach_outer
        )
        
        # Convert to class predictions
        logits = output.permute(0, 2, 3, 1)  # [B, H, W, C]
        all_logits = [out.permute(0, 2, 3, 1) for out in all_outputs]
        
        return logits, all_logits


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ARCDeepRecursiveModel(
        num_colors=10,
        base_channels=64,
        latent_channels=512,
        n_res_blocks=2
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    B, H, W = 4, 32, 32
    task_input = torch.randint(0, 12, (B, H, W), device=device)
    
    # Test regular forward
    logits = model(task_input, n_steps=3)
    print(f"Output shape: {logits.shape}")  # Should be [B, H, W, 12]
    
    # Test deep recursion
    logits, all_logits = model.deep_recursion_forward(
        task_input,
        n_inner_steps=3,
        n_outer_steps=3,
        detach_outer=True
    )
    print(f"Deep recursion output shape: {logits.shape}")
    print(f"Number of intermediate outputs: {len(all_logits)}")
    print("Model test successful!")
