"""
Encoder–decoder model for ARC-like pair transformations with direction-based loss.

Contract:
- Inputs: tensors shaped [batch, input_dim]
- Encoder maps X -> Z (latent_dim)
- Decoder maps Z -> X_hat (reconstruction)
- Direction vector for a pair (A, B): dir = B' - A'
- For same-task pairs, we maximize cosine similarity between directions.
  For different-task pairs, we minimize it (or use a margin/contrastive variant).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # Minimal MLP; extend as needed
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ArcPrizeModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


def pair_direction(z_a: torch.Tensor, z_b: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute direction vector B' - A' with optional L2 normalization per-sample."""
    d = z_b - z_a
    if normalize:
        d = F.normalize(d, p=2, dim=-1)
    return d


def direction_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    z_c: torch.Tensor,
    z_d: torch.Tensor,
    same_task: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Cosine similarity objective between two pair directions.

    If same_task is True: maximize similarity (minimize negative cosine).
    If False: minimize similarity (push apart).
    """
    dir1 = pair_direction(z_a, z_b, normalize)
    dir2 = pair_direction(z_c, z_d, normalize)
    cos_sim = F.cosine_similarity(dir1, dir2, dim=-1)
    return (-cos_sim.mean()) if same_task else (cos_sim.mean())


def contrastive_direction_loss(
    dirs_q: torch.Tensor,
    dirs_k: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE-style contrastive loss over direction vectors.

    - dirs_q: [B, D] queries
    - dirs_k: [B, D] keys
    - labels: [B] int64, label id per pair; same id => same task
    """
    dirs_q = F.normalize(dirs_q, dim=-1)
    dirs_k = F.normalize(dirs_k, dim=-1)
    logits = (dirs_q @ dirs_k.T) / temperature  # [B, B]
    # Targets: for each i, positive is the j with same label; allow multiple positives
    with torch.no_grad():
        label_eq = labels[:, None] == labels[None, :]
        mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        positives = label_eq & ~mask
        # Normalize positives per row
        pos_counts = positives.sum(dim=1, keepdim=True).clamp_min(1)
        targets = positives.float() / pos_counts
    log_probs = logits.log_softmax(dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x)


@torch.no_grad()
def predict_f(
    model: ArcPrizeModel,
    a: torch.Tensor,
    b: torch.Tensor,
    e: torch.Tensor,
    normalize_direction: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict F from (A->B) applied to E.

    Returns (z_f_pred, x_f_pred, dir_ab).
    """
    model.eval()
    z_a = model.encode(a)
    z_b = model.encode(b)
    z_e = model.encode(e)
    dir_ab = pair_direction(z_a, z_b, normalize=normalize_direction)
    z_f_pred = z_e + dir_ab
    x_f_pred = model.decode(z_f_pred)
    return z_f_pred, x_f_pred, dir_ab


# Minimal smoke test function (optional import from main)
def _smoke_test():  # pragma: no cover
    B, input_dim, latent_dim = 4, 16, 8
    model = ArcPrizeModel(input_dim, latent_dim)
    rng = torch.Generator().manual_seed(0)
    A = torch.randn(B, input_dim, generator=rng)
    Bx = torch.randn(B, input_dim, generator=rng)
    C = torch.randn(B, input_dim, generator=rng)
    Dx = torch.randn(B, input_dim, generator=rng)
    E = torch.randn(B, input_dim, generator=rng)

    z_a, a_rec = model(A)
    z_b, b_rec = model(Bx)
    z_c, c_rec = model(C)
    z_d, d_rec = model(Dx)
    loss_dir = direction_loss(z_a, z_b, z_c, z_d, same_task=True)
    loss_rec = sum(
        reconstruction_loss(x, xr)
        for x, xr in [(A, a_rec), (Bx, b_rec), (C, c_rec), (Dx, d_rec)]
    ) / 4
    z_f_pred, x_f_pred, _ = predict_f(model, A, Bx, E)
    print("loss_dir=", float(loss_dir), "loss_rec=", float(loss_rec), "pred_shape=", x_f_pred.shape)


if __name__ == "__main__":  # pragma: no cover
    _smoke_test()

# -----------------------------
# Transformer-based 2D variant
# -----------------------------

class SinCosPositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, H: int, W: int, device=None, dtype=None) -> torch.Tensor:
        """Return [1, H*W, d_model] 2D sin/cos positional encodings."""
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        y = torch.linspace(0, 1, steps=H, device=device, dtype=dtype)
        x = torch.linspace(0, 1, steps=W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H, W]
        pos = torch.stack([yy, xx], dim=-1)  # [H, W, 2]
        pos = pos.reshape(H * W, 2)
        d = self.d_model // 4
        if d == 0:
            return torch.zeros(1, H * W, self.d_model, device=device, dtype=dtype)
        dim_t = torch.arange(d, device=device, dtype=dtype)
        dim_t = (10000 ** (2 * dim_t / d))
        pe_y = pos[:, 0][:, None] / dim_t  # [HW, d]
        pe_x = pos[:, 1][:, None] / dim_t  # [HW, d]
        pe = torch.cat([torch.sin(pe_y), torch.cos(pe_y), torch.sin(pe_x), torch.cos(pe_x)], dim=1)  # [HW, 4d]
        if pe.size(1) < self.d_model:
            pad = self.d_model - pe.size(1)
            pe = torch.cat([pe, torch.zeros(pe.size(0), pad, device=device, dtype=dtype)], dim=1)
        elif pe.size(1) > self.d_model:
            pe = pe[:, : self.d_model]
        return pe.unsqueeze(0)  # [1, HW, d_model]


class ArcTransformerModel(nn.Module):
    def __init__(
        self,
        H: int,
        W: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers_enc: int = 4,
        num_layers_dec: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        *,
        input_channels: int | None = None,
        num_classes: int | None = None,
        latent_dim: int | None = None,
    ):
        """Transformer encoder-decoder for 2D grids with 2D positional encoding.

        Exactly one of input_channels or num_classes must be provided.
        - If num_classes is provided: inputs are int labels [B, H, W], reconstruction returns logits [B, H, W, C].
        - If input_channels is provided: inputs are floats [B, H, W, C], reconstruction is float with MSE.
        """
        super().__init__()
        assert (input_channels is None) ^ (num_classes is None), "Provide exactly one of input_channels or num_classes"
        self.H, self.W = H, W
        self.d_model = d_model
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.latent_dim = latent_dim or d_model

        self.pe2d = SinCosPositionalEncoding2D(d_model)

        if num_classes is not None:
            self.token_embed = nn.Embedding(num_classes, d_model)
        else:
            self.in_proj = nn.Linear(input_channels, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers_enc)
        self.to_latent = nn.Linear(d_model, self.latent_dim)

        # Decoder uses a single memory token derived from the global latent
        self.mem_from_latent = nn.Linear(self.latent_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_dec)

        out_dim = num_classes if num_classes is not None else (input_channels or 1)
        self.out_proj = nn.Linear(d_model, out_dim)

    def _seq_and_pe(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device, dtype = x.device, x.dtype
        pe = self.pe2d(self.H, self.W, device=device, dtype=torch.float32)  # [1, HW, d]
        if self.num_classes is not None:
            # x: [B, H, W] int64
            x = x.view(B, self.H * self.W)
            tok = self.token_embed(x)  # [B, HW, d]
        else:
            # x: [B, H, W, C] float
            x = x.view(B, self.H * self.W, -1)
            tok = self.in_proj(x)  # [B, HW, d]
        tok = tok + pe  # add 2D PE, stays [B, HW, d]
        return tok

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._seq_and_pe(x)
        enc = self.encoder(seq)  # [B, HW, d]
        z_global = enc.mean(dim=1)  # [B, d]
        z = self.to_latent(z_global)  # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        device = z.device
        # Prepare queries as pure positional encodings (no learned queries)
        pe = self.pe2d(self.H, self.W, device=device)  # [1, HW, d]
        queries = pe.expand(B, -1, -1)  # [B, HW, d]
        mem = self.mem_from_latent(z).unsqueeze(1)  # [B, 1, d]
        dec = self.decoder(tgt=queries, memory=mem)  # [B, HW, d]
        out = self.out_proj(dec)  # [B, HW, out_dim]
        out = out.view(B, self.H, self.W, -1)
        return out

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


def reconstruction_loss_grid_mse(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """MSE for float grids. x: [B,H,W,C], x_recon: [B,H,W,C]."""
    return F.mse_loss(x_recon, x)


def reconstruction_loss_grid_ce(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for class grids. labels: [B,H,W] int, logits: [B,H,W,C]."""
    B, H, W, C = logits.shape
    return F.cross_entropy(logits.permute(0, 3, 1, 2), labels.long())

