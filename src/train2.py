from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
# import scipy.io
# import math
# import copy
import numpy as np
import torch
import torch.nn as nn
from src.data import BandPatchDataset
from src.utils import msssim_index, sam_loss
import copy
# ----------------------- Losses & utilities -----------------------
def reconstruction_loss_mse(recon, target):
    return F.mse_loss(recon, target)

def reconstruction_loss_l1(recon, target):
    return F.l1_loss(recon, target)

def total_variation_loss_img(recon, H, W):
    n = recon.shape[0]
    imgs = recon.view(n, 1, H, W)
    tv_h = torch.abs(imgs[:,:,1:,:] - imgs[:,:,:-1,:]).mean()
    tv_w = torch.abs(imgs[:,:,:,1:] - imgs[:,:,:,:-1]).mean()
    return tv_h + tv_w

def self_expression_loss(Z, CZ):
    return F.mse_loss(Z, CZ)

def reg_C_offdiag(C, p=2):
    """
    Regularize only off-diagonal entries of C (diag should remain zero).
    p=2 -> Frobenius-like on off-diagonals; p=1 -> L1 on off-diagonals.
    """
    D = torch.diag(torch.diag(C))
    off = C - D
    if p == 1:
        return torch.norm(off, p=1)
    else:
        return torch.norm(off, p='fro')

# Proximal operator for soft-thresholding off-diagonals of C
def prox_soft_threshold_offdiag_(C_tensor, tau):
    # operate in-place on C_tensor.data
    with torch.no_grad():
        D = torch.diag(torch.diag(C_tensor.data))
        off = C_tensor.data - D
        # soft threshold
        sign = torch.sign(off)
        abs_off = torch.abs(off) - tau
        abs_off = torch.clamp(abs_off, min=0.0)
        C_tensor.data = sign * abs_off
        # ensure diag remains zero
        C_tensor.data.fill_diagonal_(0.0)

# ----------------------- Training (minibatch-safe, no in-place) -----------------------
def train_model(model, hsi: np.ndarray,
                epochs=100, batch_size=16, lr=1e-3,
                alpha=10.0, beta=1e-4, device=None,
                verbose=True, C_lr_mult=5.0, C_init_scale=1e-2,
                lambda_l1=1e-3, lambda_tv=5e-4, val_frac=0.2,
                memory_cautious=True, auto_scale_alpha=False, desired_self_ratio=1.0,
                normalize_Z_for_C=False,
                # NEW args to control C regularization and proximal step
                lambda_C_l1=0.0,        # L1 weight on off-diagonal entries (added to loss)
                use_proximal_C=False,   # if True, apply proximal operator (soft-threshold) after each opt.step
                prox_tau_scale=0.5,
                lambda_msssim: float = 1e-3,
            msssim_levels: int = 5,
            lambda_sym=1e-2, 
            lambda_sam: float =1e-3 ,   # weight for SAM loss (added to total loss)
                sam_mode:str ='cosine',
                early_stopping=True,
                patience=5,
                min_delta=1e-4,
                warmup_epochs=5):
    """
    STABLE training version
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    H, W, B = hsi.shape

    # ---------------- Dataset ----------------
    ds_full = BandPatchDataset(hsi, window_size=model.window_size)
    patches, targets = [], []
    for p, t, _ in ds_full:
        patches.append(p.unsqueeze(0))
        targets.append(t.unsqueeze(0))

    patches = torch.cat(patches).to(device)
    targets = torch.cat(targets).to(device)

    # ---------- FIXED split (deterministic) ----------
    indices = np.arange(B)
    val_idx = indices[::5]                # every 5th band
    train_idx = np.setdiff1d(indices, val_idx)
    # np.random.seed(10)
    # np.random.shuffle(indices)
    # val_frac=0.1
    # n_val = int(np.ceil(val_frac * B))
    # val_idx = indices[:n_val]
    # train_idx = indices[n_val:]
    # patches_train = patches[train_idx].to(device)
    # targets_train = targets[train_idx].to(device)
    # idxs_train = torch.from_numpy(train_idx).long()

    # patches_val = patches[val_idx].to(device)
    # targets_val = targets[val_idx].to(device)
    # idxs_val = torch.from_numpy(val_idx).long()

    train_ds = TensorDataset(
        patches[train_idx], targets[train_idx], torch.from_numpy(train_idx)
    )
    val_ds = TensorDataset(
        patches[val_idx], targets[val_idx], torch.from_numpy(val_idx)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size//2))

    # ---------------- Init C ----------------
    if model.C is None:
        model.init_C(B, device, init_scale=C_init_scale, add_learnable_scale=True)
        with torch.no_grad():
            model.log_C_scale.fill_(0.0)

    # ---------- Optimizer ----------
    other_params, C_params = [], []
    for n, p in model.named_parameters():
        if "C" in n or "log_C_scale" in n:
            C_params.append(p)
        else:
            other_params.append(p)

    opt = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": C_params, "lr": lr * C_lr_mult}
    ])

    # ---------- Warm-up: freeze C ----------
    for p in C_params:
        p.requires_grad = False

    history = {"train_mse": [], "val_mse": []}
    best_state, best_val = None, float("inf")
    wait = 0
    C_snapshots = []

    # ================= TRAIN =================
    for ep in range(epochs):
        model.train()

        if ep == warmup_epochs:
            for p in C_params:
                p.requires_grad = True
            if verbose:
                print("🔓 Unfreezing C (graph learning starts)")

        with torch.no_grad():
            Z_full = model.forward_encode_all(patches)

        train_loss = 0.0

        for p_batch, t_batch, idx_batch in train_loader:
            p_batch = p_batch.to(device)
            t_batch = t_batch.to(device)
            idx_batch = idx_batch.to(device)

            Z = model.forward_encode_all(p_batch)
            recon = model.decode_from_Z(Z)
            loss = reconstruction_loss_mse(recon, t_batch)
            loss += lambda_tv * total_variation_loss_img(recon, H, W)

            if ep >= warmup_epochs:
                CZ = model.self_expression(Z_full, idx_batch)
                loss += alpha * self_expression_loss(Z, CZ)
                loss += beta * reg_C_offdiag(model.C)
                loss += lambda_sym * torch.mean((model.C - model.C.T) ** 2)
                loss += lambda_C_l1 * reg_C_offdiag(model.C, p=1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------- Validation ----------
        model.eval()
        with torch.no_grad():
            Z_val = Z_full[val_idx]
            recon_val = model.decode_from_Z(Z_val)
            val_loss = reconstruction_loss_mse(recon_val, targets[val_idx]).item()

        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)

        if verbose:
            print(f"Epoch {ep+1:03d} | Train {train_loss:.5f} | Val {val_loss:.5f}")

        # ---------- Early stopping ----------
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if early_stopping and wait >= patience:
            print(f"🛑 Early stopping at epoch {ep+1}")
            break

        # save last C matrices
        if ep >= epochs - 5:
            C_snapshots.append(model.C.detach().cpu().numpy())

    # ---------- Restore best ----------
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- Stable C (averaged) ----------
    C_final = np.mean(C_snapshots, axis=0) if C_snapshots else model.C.detach().cpu().numpy()
    A_raw = np.abs(C_final) + np.abs(C_final.T)

    with torch.no_grad():
        Z = model.forward_encode_all(patches)
        recon = model.decode_from_Z(Z).cpu().numpy()
        recon_cube = recon.reshape(B, H, W).transpose(1, 2, 0)

    return model, Z.cpu().numpy(), C_final, A_raw, history, recon_cube
