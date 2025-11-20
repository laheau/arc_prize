import torch
from torch import nn
from typing import Optional, OrderedDict
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, num_layers = 4, d_model: int = 128, nhead: int = 8):
        super(BaseModel, self).__init__()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, d_colors: int = 512, dropout: float = 0.1, nlayer_mlp: int = 2):
        self.d_model = d_model  
        self.d_colors = d_colors
        super(TransformerEncoderLayer, self).__init__()
        self.mlp_layers = []
        for _ in range(nlayer_mlp):
            self.mlp_layers.append(nn.Conv2d(self.d_colors, self.d_colors, kernel_size=(3, 3), padding='same'))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*self.mlp_layers)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        

    def feedforward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.d_colors, 32, 32)
        x = self.mlp(x)
        return x.reshape(-1, self.d_colors, self.d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # print(src.shape)
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.feedforward(src)
        src = src + src2
        src = self.norm2(src)
        return src

class GridEncoder(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 4, nhead: int = 8, d_colors: int = 512, nlayer_mlp: int = 2):
        super(GridEncoder, self).__init__()
        self.d_model = d_model
        self.d_colors = d_colors
        self.color_embedding = nn.Conv2d(in_channels=12, out_channels=self.d_colors, kernel_size=(3, 3), padding='same')
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead, d_colors=self.d_colors, nlayer_mlp=nlayer_mlp) for _ in range(num_layers)])
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.color_embedding(src)
        output = output.reshape(-1, self.d_colors, self.d_model)
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output.mean(dim=-1)
        


class GridDecoder(nn.Module):
    def __init__(self, num_layers: int = 4, d_colors: int = 512):
        super(GridDecoder, self).__init__()
        self.d_colors = d_colors
        self.layers = []
        self.rms_norm = nn.RMSNorm(d_colors)
        for _ in range(num_layers):
            layer = []
            layer.append(nn.ConvTranspose2d(in_channels=self.d_colors, out_channels=self.d_colors, kernel_size=(2, 2), stride=(2, 2)))
            layer.append(nn.ReLU())
            layer.append(nn.Conv2d(in_channels=self.d_colors, out_channels=self.d_colors, kernel_size=(3, 3), padding='same'))
            layer.append(nn.ReLU())
            layer.append(nn.Conv2d(in_channels=self.d_colors, out_channels=self.d_colors, kernel_size=(3, 3), padding='same'))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(0.1))
            self.layers.append(nn.Sequential(*layer))
            
        self.layers = nn.ModuleList(self.layers)
        self.output_layer = nn.Conv2d(in_channels=self.d_colors, out_channels=12, kernel_size=(3, 3), padding='same')
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = src.reshape(-1, self.d_colors, 1, 1)
        for mod in self.layers:
            output = self.rms_norm((output.repeat(1, 1, 2, 2) + mod(output)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # output = mod(output)
        output = self.output_layer(output)
        return output


class UNetBlock(nn.Module):
    def __init__(self, in_channels, features, name, attention=False, z_channels=128):
        super(UNetBlock, self).__init__()
        self.z_channels = z_channels
        self.block1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding="same",
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(num_groups=features//4, num_channels=features)),
                    (name + "relu1", nn.ReLU(inplace=True))
                ]
            )
        )
        self.block2 = nn.Sequential( 
            OrderedDict(
                [
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding="same",
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(num_groups=features//4, num_channels=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.y_proj = nn.Linear(self.z_channels, features)
        self.z_proj = nn.Linear(self.z_channels, features)
        if attention:
            self.self_attention = nn.MultiheadAttention(embed_dim=features, num_heads=8, batch_first=True)

    def forward(self, x, pos_emb=None):
        x = self.block1(x)
        if pos_emb is not None:
            x = x + self.z_proj(pos_emb.permute(0, 3 ,2, 1)).permute(0, 2, 3, 1)
        x = x + self.block2(x)
        if hasattr(self, 'self_attention'):
            b, c, h, w = x.shape
            x_reshaped = x.view(b, c, h * w).permute(0, 2, 1)
            attn_output, _ = self.self_attention(x_reshaped, x_reshaped, x_reshaped)
            attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
            x = x + attn_output
        return x



class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32, layers=4, pool_kernel=(2,2)):
        super(UNet, self).__init__()
        features = init_features
        self.z_channels = 128
        self.encoder_layers = []
        self.decoder_layers = []
        for i in range(layers):
            self.encoder_layers.append(UNetBlock(in_channels if i == 0 else features * 2 ** (i - 1), features * 2 ** i, name=f"enc{i+1}", z_channels=self.z_channels))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel))
            self.decoder_layers.append(nn.ConvTranspose2d(kernel_size=pool_kernel, stride=pool_kernel, in_channels=features * 2 ** (layers - i), out_channels=features * 2 ** (layers - i - 1)))
            self.decoder_layers.append(UNetBlock((features * 2 ** (layers - i - 1)) * 2, features * 2 ** (layers - i - 1), name=f"dec{layers - i}", z_channels=self.z_channels))
        self.bottleneck = UNetBlock(features * (2 ** (layers-1)), features * (2 ** layers), name="bottleneck", attention=True, z_channels=self.z_channels)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.output_layer = nn.Conv2d(kernel_size=1, in_channels=features, out_channels=out_channels + self.z_channels)
        

    def forward(self, x, y, z):
        encs = []
        for i in range(len(self.encoder_layers)//2):
            x = self.encoder_layers[2*i](x, z)
            encs.append(x)
            x = self.encoder_layers[2*i + 1](x, z)
        x = self.bottleneck(x, z)
        for i in range(len(self.decoder_layers)//2):
            x = self.decoder_layers[2*i](x, z)
            enc = encs[-(i + 1)]
            x = torch.cat((x, enc), dim=1)
            x = self.decoder_layers[2*i + 1](x, z)

        out = self.output_layer(x)
        return out[:, :-self.z_channels, :, :], out[:, :, -self.z_channels:, :, :] 


# best_cae_32x32.py
import torch, torch.nn as nn, torch.nn.functional as F

# -----------------------------
# Building blocks
# -----------------------------
class PreActResBlock(nn.Module):
    """Pre-activation residual block (BN->ReLU->Conv)."""
    def __init__(self, ch, bottleneck=False):
        super().__init__()
        mid = ch // 2 if bottleneck else ch
        self.bn1 = nn.BatchNorm2d(ch); self.conv1 = nn.Conv2d(ch, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid); self.conv2 = nn.Conv2d(mid, ch, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out

def down_block(cin, cout, n_res=2):
    """Strided conv downsample + residual stack."""
    layers = [
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ] + [PreActResBlock(cout) for _ in range(n_res)]
    return nn.Sequential(*layers)

def up_block(cin, cout, n_res=2):
    """Nearest-neighbor upsample + conv (resize-conv) + residual stack (avoids checkerboards)."""
    layers = [
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ] + [PreActResBlock(cout) for _ in range(n_res)]
    return nn.Sequential(*layers)

# -----------------------------
# Model
# -----------------------------
class BestCAE32(nn.Module):
    def __init__(self, in_ch=3, base=64, out_ch=3, n_res=2):
        super().__init__()
        # Encoder
        self.enc1 = down_block(base, base*2, n_res)   # 32 -> 16
        self.enc2 = down_block(base*2, base*4, n_res) # 16 -> 8
        self.enc3 = down_block(base*4, base*8, n_res) # 8 -> 4 (bottleneck channels)

        # Bottleneck residual stack
        self.bottleneck = nn.Sequential(*[PreActResBlock(base*8) for _ in range(n_res)])

        # Decoder (mirror) with U-Net skips
        self.dec3 = up_block(base*8, base*4, n_res)   # 4 -> 8
        self.dec2 = up_block(base*8, base*2, n_res)   # 8 -> 16  (concat skip)
        self.dec1 = up_block(base*4, base, n_res)     # 16 -> 32 (concat skip)

        self.head = nn.Conv2d(base*2, base, 1)      # concat with stem skip
        self.out_act = nn.Sigmoid()

        self.task_head = nn.Conv2d(24, base, 1)
        self.inference_head = nn.Conv2d(12, base, 1)
        self.answer_head = nn.Conv2d(base, 12, 1)
        self.q_head = nn.Conv2d(base, 1, 1)
        
        self.task_input_proj = nn.Conv2d(32, base, 1)
        self.task_output_proj = nn.Conv2d(base, 32, 1)

        # Lecun initialization
        self.z_init_task = nn.Parameter(torch.empty(1, base, 32, 32))
        self.y_init_task = nn.Parameter(torch.empty(1, base, 32, 32))

        self.z_init_solution = nn.Parameter(torch.empty(1, base, 32, 32))
        self.y_init_solution = nn.Parameter(torch.empty(1, base, 32, 32))
        nn.init.kaiming_normal_(self.z_init_task)
        nn.init.kaiming_normal_(self.y_init_task)
        nn.init.kaiming_normal_(self.z_init_solution)
        nn.init.kaiming_normal_(self.y_init_solution)
        
    def encode(self, x):
        s0 = x
        s1 = self.enc1(s0)  # 16x16, 2b
        s2 = self.enc2(s1)  # 8x8,   4b
        s3 = self.enc3(s2)  # 4x4,   8b
        z  = self.bottleneck(s3)
        return z, (s0, s1, s2)

    def decode(self, z, skips):
        s0, s1, s2 = skips
        y = self.dec3(z)                 # 4->8, ch=4b
        y = torch.cat([y, s2], dim=1)    # 8x8, ch=8b
        y = self.dec2(y)                 # 8->16, ch=2b
        y = torch.cat([y, s1], dim=1)    # 16x16, ch=4b
        y = self.dec1(y)                 # 16->32, ch=b
        y = torch.cat([y, s0], dim=1)    # 32x32, ch=2b
        return self.head(y)

    def forward(self, x):
        z, skips = self.encode(x)
        x_hat = self.decode(z, skips)
        return x_hat, z

    def get_task(self, x):
        task_feat = self.task_head(x)
        return task_feat 
    
    def get_inference(self, x):
        inference_feat = self.inference_head(x)
        return inference_feat
    
    # def z_init(self):
    #     return torch.zeros((1, 512, 4, 4), device=next(self.parameters()).device)


    def latent_recursion(self, x, y, z, n=3, task = None):
        for _ in range(n):
            z, _ = self.forward(x + y + z + (self.task_input_proj(task) if task is not None else 0))
        y, _ = self.forward(y + z)
        return z, y
    
    
    def deep_recursion(self, x, y, z, n=6, T=3, task = None):
        with torch.no_grad():
            for j in range(T-1):
                z, y = self.latent_recursion(x, y, z, n=n, task=task)
        z, y = self.latent_recursion(x, y, z, n=n, task=task)

        answ = self.task_output_proj(y) if task is None else self.answer_head(y)
        return (z.detach(), y.detach()), answ, self.q_head(y)
    

