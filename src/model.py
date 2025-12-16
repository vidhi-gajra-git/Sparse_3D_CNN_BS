# how can i make the model better ? it is working good with accuracy as discussed but is there a way to make the architecture more deeper / better
"""
Hybrid 3D-CNN + (optional) attention + Self-Expression dictionary learning
Extended with visualization utilities: epoch-wise loss plotting, per-band RMSE, SNR computation and visualization of high-noise bands.
"""
# lambda_msssim: float = 1e-3

# msssim_levels: int = 5
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.io




# ----------------------- Models -----------------------
class Conv3DEncoder(nn.Module):
    def __init__(self, in_spectral, spatial_H, spatial_W, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.Tanh(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(8,16,kernel_size=(3,3,3), padding=(1,1,1)),
            nn.Tanh(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(16,32,kernel_size=(3,3,3), padding=(1,1,1)),
            nn.Tanh(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(32,64,kernel_size=(3,3,3), padding=(1,1,1)),
            nn.Tanh(),
            nn.AdaptiveAvgPool3d((None,1,1))
        )
        self.fc = nn.Linear(64 * in_spectral, latent_dim)
        self.post_fc_ln = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)             # -> (N,1,S,H,W)
        x = self.enc(x)               # -> (N, C, S', 1, 1)
        b, ch, s, _, _ = x.shape
        x = x.view(b, ch * s)
        z = self.fc(x)
        if hasattr(self, 'post_fc_ln'):
            z = self.post_fc_ln(z)
        return z


class DecoderFromLatent(nn.Module):
    def __init__(self, latent_dim, out_dim, dropout=0.2, hidden_mult=3):
        super().__init__()
        h = max(latent_dim * hidden_mult, 256)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h//2, h//4),
            nn.Linear(h//4, out_dim),
            nn.GELU(),
        )
    def forward(self, z):
        return self.net(z)

class SpectralAttention(nn.Module):
    def __init__(self, dim, nheads=4, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=nheads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(inplace=True), nn.Linear(dim*2, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, Z):
        # Z: (B, dim)
        seq = Z.unsqueeze(0)  # (1, B, dim)
        attn_out, _ = self.mha(seq, seq, seq)
        seq = self.ln1(seq + attn_out)
        ff_out = self.ff(seq)
        seq = self.ln2(seq + ff_out)
        return seq.squeeze(0)

class HybridModel(nn.Module):
    def __init__(self, window_size, H, W, latent_dim=64, use_attention=False,
                 decoder_dropout=0.2, decoder_hidden_mult=3):
        super().__init__()
        self.window_size = window_size
        self.H = H; self.W = W
        self.latent_dim = latent_dim
        self.encoder = Conv3DEncoder(in_spectral=window_size, spatial_H=H, spatial_W=W, latent_dim=latent_dim)
        self.decoder = DecoderFromLatent(latent_dim=latent_dim, out_dim=H*W,
                                         dropout=decoder_dropout, hidden_mult=decoder_hidden_mult)
        self.use_attention = use_attention
        if use_attention:
            self.attn = SpectralAttention(dim=latent_dim)
        # C and log_C_scale will be initialized by init_C
        self.C = None
        self.log_C_scale = None

    def init_C(self, num_bands, device, init_scale=1e-2, add_learnable_scale=True):
        """
        Initialize C on given device.
        Use init_scale larger than 1e-3 (1e-2 or 1e-1 recommended).
        add_learnable_scale -> create log_C_scale parameter so optimizer can amplify C.
        """
        C = init_scale * torch.randn((num_bands, num_bands), dtype=torch.float32, device=device)
        C = C - torch.diag(torch.diag(C))
        self.C = nn.Parameter(C)
        if add_learnable_scale:
            # log scale initialized at a small positive value so C isn't tiny
            self.log_C_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
        else:
            self.log_C_scale = None

    def forward_encode_all(self, patches):
        device = next(self.parameters()).device
        patches = patches.to(device)
        Z = self.encoder(patches)        # (B, latent_dim)
        if self.use_attention:
            Z = self.attn(Z)             # (B, latent_dim)
        return Z

    def decode_from_Z(self, Z):
        return self.decoder(Z)

    def self_expression(self, Z_full, global_idxs=None, normalize_Z=False, eps=1e-8):
        """
        Z_full: (B, D)
        If global_idxs is None -> returns (B, D) as C_masked @ Z_full
        Else returns (N, D) for rows corresponding to global_idxs: C_masked[global_idxs] @ Z_full
        Uses a masked copy and multiplies by exp(log_C_scale) to allow learnable amplification.
        """
        if getattr(self, 'log_C_scale', None) is not None:
            C_scale = torch.exp(self.log_C_scale)
        else:
            C_scale = 1.0

        # masked copy (no in-place)
        C = self.C
        diag = torch.diag(torch.diag(C))
        C_masked = C - diag           # new tensor tied to autograd

        if normalize_Z:
            Z_norm = torch.norm(Z_full, p=2, dim=1, keepdim=True).clamp_min(eps)
            Z_proc = Z_full / Z_norm
        else:
            Z_proc = Z_full

        target_device = Z_proc.device

        if global_idxs is None:
            if C_masked.device != target_device:
                C_rows = C_masked.to(target_device)
            else:
                C_rows = C_masked
            C_rows = C_scale * C_rows
            return torch.matmul(C_rows, Z_proc)
        else:
            if not isinstance(global_idxs, torch.Tensor):
                idx_for_C = torch.as_tensor(global_idxs, dtype=torch.long, device=C_masked.device)
            elif global_idxs.device != C_masked.device:
                idx_for_C = global_idxs.to(C_masked.device)
            else:
                idx_for_C = global_idxs
            C_subset = C_masked[idx_for_C]            # (N, B) on C.device
            if C_subset.device != target_device:
                C_subset = C_subset.to(target_device, non_blocking=True)
            C_subset = C_scale * C_subset
            return torch.matmul(C_subset, Z_proc)     # (N, D)

