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
                alpha=1.0, beta=1e-4, device=None,
                verbose=True, C_lr_mult=5.0, C_init_scale=1e-2,
                lambda_l1=1e-3, lambda_tv=5e-4, val_frac=0.2,
                memory_cautious=True, auto_scale_alpha=True, desired_self_ratio=1.0,
                normalize_Z_for_C=False,
                # NEW args to control C regularization and proximal step
                lambda_C_l1=0.0,        # L1 weight on off-diagonal entries (added to loss)
                use_proximal_C=False,   # if True, apply proximal operator (soft-threshold) after each opt.step
                prox_tau_scale=0.5,
                lambda_msssim: float = 1e-3,
msssim_levels: int = 5,
lambda_sam: float =1e-3 ,   # weight for SAM loss (added to total loss)
    sam_mode:str ='cosine',
    early_stopping=True,
    patience=5,
    min_delta=1e-4):    # relative to exp(log_C_scale) to compute tau
    """
    Returns: (model, Z_full_np, C_final_np, A_raw_np, history, recon_cube)
    history is a dict with epoch-wise metrics.
    recon_cube is (H,W,B) numpy array of reconstructed images.
    """

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    H, W, B = hsi.shape
    print(H,W,B)
    ds_full = BandPatchDataset(hsi, window_size=model.window_size)

    # Build full arrays (B is typically small enough)
    patches = []
    targets = []
    for p, t, idx in ds_full:
        patches.append(p.unsqueeze(0))   # (1, S, H, W)
        targets.append(t.unsqueeze(0))   # (1, H*W)
    patches = torch.cat(patches, dim=0)  # (B, S, H, W)
    targets = torch.cat(targets, dim=0)  # (B, H*W)

    # train/val split on band indices
    indices = np.arange(B)
    np.random.seed(10)
    np.random.shuffle(indices)
    n_val = int(np.ceil(val_frac * B))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    patches_train = patches[train_idx].to(device)
    targets_train = targets[train_idx].to(device)
    idxs_train = torch.from_numpy(train_idx).long()

    patches_val = patches[val_idx].to(device)
    targets_val = targets[val_idx].to(device)
    idxs_val = torch.from_numpy(val_idx).long()

    train_ds = TensorDataset(patches_train, targets_train, idxs_train)
    val_ds = TensorDataset(patches_val, targets_val, idxs_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size//2), shuffle=False)

    # init C on same device as model (to avoid frequent moves)
    if model.C is None:
        print(C_init_scale,type(C_init_scale)
        model.init_C(num_bands=int(B), device=device, init_scale=C_init_scale, add_learnable_scale=True)
        # set starting scale to modest value so C isn't tiny
        with torch.no_grad():
            # e.g. start with scale 2.0 (exp(log)=2)
            model.log_C_scale.data.fill_(torch.log(torch.tensor(2.0, dtype=torch.float32, device=device)))

    # optimizer param groups (C gets higher lr)
    named = list(model.named_parameters())
    C_params = [p for n, p in named if n == "C" or n.endswith(".C")]
    logC_params = [p for n, p in named if 'log_C_scale' in n]
    other_params = [p for n, p in named if (n != "C" and not n.endswith(".C") and 'log_C_scale' not in n)]
    param_groups = [{'params': other_params, 'lr': lr}]
    if len(C_params) > 0:
        param_groups.append({'params': C_params, 'lr': lr * float(C_lr_mult)})
    if len(logC_params) > 0:
        param_groups.append({'params': logC_params, 'lr': lr * float(C_lr_mult)})

    opt = torch.optim.Adam(param_groups, lr=lr)
     

    # container for epoch history
    history = {"train_mse":[], "val_mse":[], "train_l1":[], "val_l1":[], "train_tv":[], "val_tv":[],
               "train_self":[], "val_self":[], "train_reg":[], "val_reg":[], "train_sam":[], "val_sam":[],
               "train_msssim":[], "val_msssim":[]}

    # Auto-scale alpha by comparing small-sample recon vs self magnitudes
    if auto_scale_alpha:
        model.eval()
        with torch.no_grad():
            all_patches_dev = patches.to(device)
            Z_full_probe = model.forward_encode_all(all_patches_dev)
            # small reconstruction estimate
            sample_n = min(16, patches_train.size(0))
            sample_p = patches_train[:sample_n]
            sample_t = targets_train[:sample_n]
            Z_sample = model.forward_encode_all(sample_p)

            # use corresponding global indices for the sample patch rows so CZ_sample shape matches Z_sample
            sample_global_idxs = train_idx[:sample_n]
            sample_global_idxs_t = torch.from_numpy(np.asarray(sample_global_idxs)).long().to(device)

            recon_sample = model.decode_from_Z(Z_sample)
            CZ_sample = model.self_expression(Z_full_probe, global_idxs=sample_global_idxs_t, normalize_Z=normalize_Z_for_C)
            recon_loss = float(reconstruction_loss_mse(recon_sample, sample_t).item())
            self_loss = float(self_expression_loss(Z_sample, CZ_sample).item())
            ratio = recon_loss / (self_loss + 1e-12)
            
            alpha = float(max(1e-6, desired_self_ratio * ratio))
            # alpha = float(np.clip(alpha, 1e-3, 50.0))
            if verbose:
                print(f"[auto-scale] estimated recon={recon_loss:.6e}, self={self_loss:.6e}, set alpha={alpha:.4e}")
    # best_val = float("inf")
    best_state = None
    wait = 0
    best_val = 1e9
    for ep in range(epochs):
        model.train()
        # compute full-band Z once per epoch
        all_patches = patches.to(device)
        if memory_cautious:
            with torch.no_grad():
                Z_full = model.forward_encode_all(all_patches).detach()
        else:
            Z_full = model.forward_encode_all(all_patches)

        running = {"mse":0.0, "l1":0.0, "tv":0.0, "self":0.0, "reg":0.0, "count":0,"sam":0.0,"msssim":0.0}
        num_batches = len(train_loader)
        for bidx, (p_batch, t_batch, idx_batch) in enumerate(train_loader):
            p_batch = p_batch.to(device)
            t_batch = t_batch.to(device)
            idx_batch = idx_batch.to(device)

            # encode minibatch for reconstruction
            Z_batch = model.forward_encode_all(p_batch)   # (N_batch, D)

            # compute CZ for minibatch using masked C (no in-place)
            CZ_batch = model.self_expression(Z_full, global_idxs=idx_batch, normalize_Z=normalize_Z_for_C)  # (N_batch, D)

            recon_batch = model.decode_from_Z(Z_batch)  # (N_batch, H*W)
            # reshape to images (N,1,H,W) for SSIM/MS-SSIM
            recon_img = recon_batch.view(recon_batch.shape[0], 1, H, W)
            target_img = t_batch.view(t_batch.shape[0], 1, H, W)

            # compute MS-SSIM (returns scalar tensor in (0,1])
            loss_msssim_val = 1.0 - msssim_index(recon_img, target_img, levels=msssim_levels, window_size=11, sigma=1.5, L=1.0)

            loss_mse = reconstruction_loss_mse(recon_batch, t_batch)
            loss_l1 = reconstruction_loss_l1(recon_batch, t_batch)
            loss_tv = total_variation_loss_img(recon_batch, H, W)
            loss_self = self_expression_loss(Z_batch, CZ_batch)
            loss_sam = sam_loss(recon_batch, t_batch, mode=sam_mode)

            # regularize off-diagonals only:
            loss_reg = reg_C_offdiag(model.C, p=2)
            # optional L1 on off-diagonals for sparsity
            loss_C_l1 = lambda_C_l1 * reg_C_offdiag(model.C, p=1) if lambda_C_l1 > 0 else 0.0
            loss = loss_mse + (float(lambda_l1) * loss_l1) + (float(lambda_tv) * loss_tv) \
                   + (float(alpha) * loss_self) + (float(beta) * loss_reg) \
                   + (loss_C_l1 if isinstance(loss_C_l1, torch.Tensor) else torch.tensor(float(loss_C_l1), device=loss_mse.device))

            if lambda_sam is not None and float(lambda_sam) != 0.0:
                loss_sam = sam_loss(recon_batch, t_batch, mode=sam_mode)
                loss = loss + float(lambda_sam) * loss_sam

            if lambda_msssim is not None and float(lambda_msssim) != 0.0:
                loss = loss + float(lambda_msssim) * loss_msssim_val

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            # proximal update (soft threshold) on off-diagonals if requested
            if use_proximal_C and getattr(model, 'C', None) is not None:
                # compute tau relative to scale of C
                C_scale_val = float(torch.exp(model.log_C_scale).detach().cpu().numpy()) if getattr(model, 'log_C_scale', None) is not None else 1.0
                tau = prox_tau_scale * (C_scale_val * 1e-3)  # small absolute threshold; you can tune this
                prox_soft_threshold_offdiag_(model.C, tau)
            # safe optimizer step + soft clamps to keep C healthy
            


            running["mse"] += float(loss_mse.item()) * p_batch.size(0)
            running["l1"] += float(loss_l1.item()) * p_batch.size(0)
            running["tv"] += float(loss_tv.item()) * p_batch.size(0)
            running["self"] += float(loss_self.item()) * p_batch.size(0)
            running["reg"] += float(loss_reg.item()) * p_batch.size(0)
            running["sam"] += float(loss_sam.item()) * p_batch.size(0)
            running["msssim"] = running.get("msssim", 0.0) + float(loss_msssim_val.item()) * p_batch.size(0)

            running["count"] += p_batch.size(0)

        # epoch stats
        train_mse = running["mse"] / max(1, running["count"])
        train_l1 = running["l1"] / max(1, running["count"])
        train_tv = running["tv"] / max(1, running["count"])
        train_self = running["self"] / max(1, running["count"])
        train_reg = running["reg"] / max(1, running["count"])
        train_sam = running["sam"] / max(1, running["count"])
        train_msssim = running["msssim"] / max(1, running["count"])

        # eval on val
        model.eval()
        with torch.no_grad():
            Z_full_eval = model.forward_encode_all(all_patches)
            CZ_val_full = model.self_expression(Z_full_eval, normalize_Z=normalize_Z_for_C)
            CZ_val = CZ_val_full[val_idx]
            Z_val = Z_full_eval[val_idx]
            recon_val = model.decode_from_Z(Z_val)
            t_val = targets[val_idx].to(device)
            recon_val_img = recon_val.view(recon_val.shape[0], 1, H, W)
            t_val_img = t_val.view(t_val.shape[0], 1, H, W)
            val_msssim = float(msssim_index(recon_val_img, t_val_img, levels=msssim_levels).item())

            val_mse = float(reconstruction_loss_mse(recon_val, t_val).item())
            val_l1 = float(reconstruction_loss_l1(recon_val, t_val).item())
            val_tv = float(total_variation_loss_img(recon_val, H, W).item())
            val_self = float(self_expression_loss(Z_val, CZ_val).item())
            val_reg = float(reg_C_offdiag(model.C, p=2).item())
            val_sam = float(sam_loss(recon_val.detach(), t_val.detach(), mode=sam_mode).item())

            val_diff = (recon_val - t_val).view(recon_val.shape[0], -1)
            per_band_rmse_val = torch.sqrt((val_diff**2).mean(dim=1)).cpu().numpy()

            c_abs = model.C.detach().abs().cpu().numpy()
            c_max = c_abs.max() if c_abs.size else 0.0
            c_mean = c_abs.mean() if c_abs.size else 0.0
            C_scale_val = float(torch.exp(model.log_C_scale).detach().cpu().numpy()) if getattr(model, 'log_C_scale', None) is not None else 1.0

        # store history
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['train_l1'].append(train_l1)
        history['val_l1'].append(val_l1)
        history['train_tv'].append(train_tv)
        history['val_tv'].append(val_tv)
        history['train_self'].append(train_self)
        history['val_self'].append(val_self)
        history['train_reg'].append(train_reg)
        history['val_reg'].append(val_reg)
        history['train_sam'].append(train_sam)
        history['val_sam'].append(val_sam)
        history['train_msssim'].append(train_msssim)
        history['val_msssim'].append(val_msssim)
        current_val = history["val_mse"][-1]
        if verbose and (ep % max(1, epochs//20) == 0 or ep == epochs-1):
            print(f"Epoch {ep+1}/{epochs} | train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f} | train L1: {train_l1:.6f} | val L1: {val_l1:.6f} | train TV: {train_tv:.6f} | val TV: {val_tv:.6f} | train SAM: {train_sam:.6f} | val SAM: {val_sam:.6f}| train SSIM: {train_msssim :.6f} | val SSIM: {val_msssim:.6f}")
            print(f"   train_self: {train_self:.6f} | val_self: {val_self:.6f} | reg: {train_reg:.6f} | max|C|: {c_max:.6e} | C_scale: {C_scale_val:.4e}")
            print("   per-band RMSE (val first 8):", np.round(per_band_rmse_val[:8], 6))

        if val_mse < best_val:
            best_val = val_mse
            # optionally save best model state here
        if early_stopping:
            if current_val < best_val - min_delta:
                best_val = current_val
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep+1}/{epochs}. Best val MSE: {best_val:.6f}")
                break

    # final Z, C, A
    model.eval()
    with torch.no_grad():
        all_patches = patches.to(device)
        Z = model.forward_encode_all(all_patches)
        C_final = model.C.detach().cpu().numpy()
        # recon for all bands
        recon_flat = model.decode_from_Z(Z).detach().cpu().numpy()  # (B, H*W)
        recon_cube = recon_flat.reshape(-1, H, W).transpose(1,2,0)  # (H, W, B)
        A_raw = np.abs(C_final) + np.abs(C_final.T)
        A_display = A_raw / (A_raw.max() + 1e-12)
        band_scores = np.max(A_raw, axis=1)
        band_scores = np.clip(band_scores, a_min=0.0, a_max=None)
        if band_scores.sum() == 0:
            band_imp = np.ones_like(band_scores) / float(len(band_scores))
        else:
            band_imp = band_scores / float(band_scores.sum())

        print("A_raw stats: min, mean, max ->", float(A_raw.min()), float(A_raw.mean()), float(A_raw.max()))
        print("A_display stats (scaled to [0,1]): min, mean, max ->", float(A_display.min()), float(A_display.mean()), float(A_display.max()))
        print("band_imp stats: min, mean, max ->", float(band_imp.min()), float(band_imp.mean()), float(band_imp.max()))
    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    return model, Z.detach().cpu().numpy(), C_final, A_raw, history, recon_cube

