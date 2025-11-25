import torch
import torch.nn as nn
import math

class SinCosPositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        
        # Precompute positional encodings
        pe = torch.zeros(max_h * max_w, d_model)
        position_h = torch.arange(max_h).unsqueeze(1).repeat(1, max_w).flatten().unsqueeze(1)
        position_w = torch.arange(max_w).unsqueeze(0).repeat(max_h, 1).flatten().unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        
        # Height encoding
        pe[:, 0:d_model//2:2] = torch.sin(position_h * div_term)
        pe[:, 1:d_model//2:2] = torch.cos(position_h * div_term)
        
        # Width encoding
        pe[:, d_model//2::2] = torch.sin(position_w * div_term)
        pe[:, d_model//2+1::2] = torch.cos(position_w * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, HW, D]

    def forward(self, x):
        # x: [B, HW, D]
        return x + self.pe[:, :x.size(1), :]

class VARCModel(nn.Module):
    """
    Vision ARC (VARC) Model.
    Treats ARC as an image-to-image translation problem using a Transformer.
    Input: Grid (H, W) -> Flattened -> Embedding + PosEnc -> Transformer -> Logits -> Grid
    """
    def __init__(
        self,
        canvas_size=32,
        num_colors=12, # 0-9 colors, + padding/indicators
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.seq_len = canvas_size * canvas_size
        
        self.embedding = nn.Embedding(num_colors, d_model)
        self.pos_encoding = SinCosPositionalEncoding2D(d_model, canvas_size, canvas_size)
        
        # We use a Transformer Encoder-only architecture for "image-to-image" 
        # (like BERT/ViT-MAE but predicting all tokens)
        # Or Encoder-Decoder. The paper mentions "vanilla Vision Transformer (ViT)" which is usually Encoder.
        # However, for image-to-image, a Decoder is often useful or just a dense head on Encoder.
        # Let's use a strong Encoder-Decoder structure as it's more general for translation.
        # Actually, for fixed size grid-to-grid, an Encoder with a dense head is sufficient and faster.
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-LN is generally better
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, num_colors)
        
    def forward(self, x):
        """
        x: [B, H, W] integer tensor
        """
        B, H, W = x.shape
        x_flat = x.view(B, H * W)
        
        # Embed
        x_emb = self.embedding(x_flat) # [B, L, D]
        
        # Add Positional Encoding
        x_emb = self.pos_encoding(x_emb)
        
        # Transformer
        features = self.transformer(x_emb) # [B, L, D]
        
        # Prediction
        logits = self.head(features) # [B, L, num_colors]
        logits = logits.view(B, H, W, -1)
        
        return logits
