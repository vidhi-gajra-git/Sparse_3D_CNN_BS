import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ----------------------- Loss helpers -----------------------

def reconstruction_loss_mse(recon, target):
    return F.mse_loss(recon, target)

def total_variation_loss_img(recon, H, W):
    imgs = recon.view(recon.shape[0], 1, H, W)
    return (
        torch.abs(imgs[:, :, 1:, :] - imgs[:, :, :-1, :]).mean() +
        torch.abs(imgs[:, :, :, 1:] - imgs[:, :, :, :-1]).mean()
    )

def self_expression_loss(Z, CZ):
    return F.mse_loss(Z, CZ)

def reg_C_offdiag(C):
    D = torch.diag(torch.diag(C))
    return torch.norm(C - D, p='fro')

def symmetry_loss(C):
    return torch.mean((C - C.T) ** 2)

# ----------------------- Stable Training -----------------------

def train_model(
    model,
    hsi,
    epochs=40,
    batch_size=16,
    lr=1e-3,
    device=None,

    # --- STABILITY PARAMETERS ---
    alpha=5.0,                  # FIXED (no autoscale)
    beta=1e-4,
    lambda_tv=2e-4,
    lambda_msssim=5e-4,
    lambda_sym=1e-2,
    lambda_C_l1=1e-3,

    C_lr_mult=2.0,
    C_init_scale=1e-1,

    warmup_epochs=5,
    early_stopping=True,
    patience=6,
    min_delta=1e-4,

    verbose=True
):
    """
    Stable two-stage training for HybridModel.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    H, W, B = hsi.shape

    # ------------------ Dataset ------------------
    ds = BandPatchDataset(hsi, window_size=model.window_size)

    patches, targets = [], []
    for p, t, _ in ds:
        patches.append(p.unsqueeze(0))
        targets.append(t.unsqueeze(0))

    patches = torch.cat(patches).to(device)     # (B, S, H, W)
    targets = torch.cat(targets).to(device)     # (B, H*W)

    # ------------------ DETERMINISTIC SPLIT ------------------
    indices = np.arange(B)
    val_idx = indices[::5]
    train_idx = np.setdiff1d(indices, val_idx)

    train_ds = TensorDataset(
        patches[train_idx], targets[train_idx], torch.from_numpy(train_idx)
    )
    val_ds = TensorDataset(
        patches[val_idx], targets[val_idx], torch.from_numpy(val_idx)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size // 2))

    # ------------------ Init C ------------------
    if model.C is None:
        model.init_C(B, device, init_scale=C_init_scale, add_learnable_scale=True)
        with torch.no_grad():
            model.log_C_scale.fill_(0.0)

    # ------------------ Optimizer ------------------
    params = [
        {"params": [p for n, p in model.named_parameters() if "C" not in n], "lr": lr},
        {"params": [model.C, model.log_C_scale], "lr": lr * C_lr_mult}
    ]
    opt = torch.optim.Adam(params)

    # ------------------ Warmup ------------------
    for p in model.C.parameters():
        p.requires_grad = False

    best_val = float("inf")
    best_state = None
    wait = 0
    C_history = []

    history = {"train_mse": [], "val_mse": []}

    # ================== TRAIN ==================
    for ep in range(epochs):

        if ep == warmup_epochs:
            for p in model.C.parameters():
                p.requires_grad = True
            if verbose:
                print("🔓 C unfrozen — starting graph learning")

        model.train()
        Z_full = model.forward_encode_all(patches).detach()

        running_loss = 0.0

        for p_batch, t_batch, idx_batch in train_loader:
            p_batch = p_batch.to(device)
            t_batch = t_batch.to(device)
            idx_batch = idx_batch.to(device)

            Z = model.forward_encode_all(p_batch)
            recon = model.decode_from_Z(Z)
            CZ = model.self_expression(Z_full, idx_batch)

            loss = reconstruction_loss_mse(recon, t_batch)

            if ep >= warmup_epochs:
                loss += alpha * self_expression_loss(Z, CZ)
                loss += beta * reg_C_offdiag(model.C)
                loss += lambda_sym * symmetry_loss(model.C)

            loss += lambda_tv * total_variation_loss_img(recon, H, W)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ------------------ VALIDATION ------------------
        model.eval()
        with torch.no_grad():
            Z_val = Z_full[val_idx]
            recon_val = model.decode_from_Z(Z_val)
            val_loss = reconstruction_loss_mse(recon_val, targets[val_idx]).item()

        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)

        if verbose:
            print(f"Epoch {ep+1:02d} | Train MSE {train_loss:.5f} | Val MSE {val_loss:.5f}")

        # ------------------ EARLY STOP ------------------
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if early_stopping and wait >= patience:
            if verbose:
                print(f"🛑 Early stopping at epoch {ep+1}")
            break

        # save last C matrices
        if ep >= epochs - 5:
            C_history.append(model.C.detach().cpu().numpy())

    # ------------------ FINALIZE ------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    C_final = np.mean(C_history, axis=0) if C_history else model.C.detach().cpu().numpy()
    A_raw = np.abs(C_final) + np.abs(C_final.T)

    with torch.no_grad():
        Z = model.forward_encode_all(patches)
        recon = model.decode_from_Z(Z).cpu().numpy()
        recon_cube = recon.reshape(B, H, W).transpose(1, 2, 0)

    return model, Z.cpu().numpy(), C_final, A_raw, history, recon_cube
