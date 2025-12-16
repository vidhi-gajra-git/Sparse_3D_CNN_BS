import scipy.io
# import math
# import copy

from sklearn.metrics import accuracy_score, cohen_kappa_score,confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# import pandas as pd
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import SpectralClustering
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
# ----------------------- Helpers (load, selection, testing) -----------------------
def load_hsi(mat_path, gt_path):
    data = scipy.io.loadmat(mat_path)
    gt = scipy.io.loadmat(gt_path)
    data_key = sorted(k for k in data.keys() if not k.startswith('__'))[-1]
    gt_key = sorted(k for k in gt.keys() if not k.startswith('__'))[-1]

    cube = np.array(data[data_key], dtype=np.float32)
    labels = np.array(gt[gt_key], dtype=np.int64)

    # Normalize cube globally to 0-1
    cube = (cube - cube.min()) / (cube.max() - cube.min() + 1e-12)

    mask = labels > 0
    X = cube[mask]           # spectral vectors of pixels
    Y = labels[mask] - 1     # zero-based labels

    return cube, X, Y

def compute_metrics(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    aa = class_acc.mean()

    kappa = cohen_kappa_score(y_true, y_pred)

    return oa, aa, kappa
def split_XY_train_val_test(X, Y, train_size=0.6, val_size=0.2, test_size=0.2, seed=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train vs temp
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=seed
    )
    train_idx, temp_idx = next(sss1.split(X_scaled, Y))

    X_train, Y_train = X_scaled[train_idx], Y[train_idx]
    X_temp, Y_temp = X_scaled[temp_idx], Y[temp_idx]

    # Val vs Test
    val_ratio = val_size / (val_size + test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=val_ratio,
        random_state=seed
    )
    val_idx, test_idx = next(sss2.split(X_temp, Y_temp))

    X_val, Y_val = X_temp[val_idx], Y_temp[val_idx]
    X_test, Y_test = X_temp[test_idx], Y_temp[test_idx]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def split_XY(X, Y, test_size=0.2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=test_size, random_state=42, stratify=Y)
    return X_train, X_test, Y_train, Y_test

def get_band_importance_from_dict(A, method='l1'):
    A = np.asarray(A)
    B = A.shape[0]
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        scores = np.sum(np.abs(A), axis=1)
    else:
        if method == 'l1':
            scores = np.sum(np.abs(A), axis=1)
        elif method == 'l2':
            scores = np.sqrt(np.sum(A**2, axis=1))
        elif method == 'max':
            scores = np.max(np.abs(A), axis=1)
        elif method == 'sum':
            scores = np.sum(A, axis=1)
        else:
            raise ValueError("Unknown method for importance")

    scores = np.clip(scores, a_min=0.0, a_max=None)
    if scores.sum() == 0:
        return np.ones(B, dtype=float) / float(B)
    return scores / float(scores.sum())

def select_topk_bands_from_A(A, k=10):
    band_scores = np.sum(np.abs(A), axis=1)
    top_k_bands = np.argsort(band_scores)[-k:][::-1]
    return top_k_bands


def compute_band_scores_all(A, k=20):
    """
    Returns:
        dict: method_name -> top-k band indices
    """
    A = np.asarray(A)
    B = A.shape[0]
    eps = 1e-12

    # ----- Score vectors -----
    L1 = np.sum(np.abs(A), axis=1)
    L2 = np.sqrt(np.sum(A*A, axis=1))
    MAX = np.max(np.abs(A), axis=1)

    U, S, Vt = np.linalg.svd(A)
    SVD = np.abs(U[:,0])

    G = nx.from_numpy_array(np.abs(A))
    ev = nx.eigenvector_centrality_numpy(G, weight='weight')
    EIG = np.array([ev[i] for i in range(B)])

    pr = nx.pagerank(G, alpha=0.85, weight='weight')
    PR = np.array([pr[i] for i in range(B)])

    # ----- Top-k indices -----
    results = {
        "L1"                   : np.argsort(L1)[-k:][::-1],
        "L2"                   : np.argsort(L2)[-k:][::-1],
        "MAX"                  : np.argsort(MAX)[-k:][::-1],
        "SVD"                  : np.argsort(SVD)[-k:][::-1],
        "EigenvectorCentrality": np.argsort(EIG)[-k:][::-1],
        "Pagerank"             : np.argsort(PR)[-k:][::-1],
    }
    return results

def select_topk_bands_multi(A, k=10, use_spectral_clusters=True):
    """
    Returns a dictionary:
        ranking_method -> top-k band indices
    """
    scores = compute_band_scores_all(A)
    results = {}

    # Top-k for each method
    for name, score_vec in scores.items():
        topk = np.argsort(score_vec)[-k:][::-1]
        results[name] = topk

    # Optional: spectral clustering
    if use_spectral_clusters:
        try:
            sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
            labels = sc.fit_predict(A)
            reps = []
            for ci in range(k):
                idx = np.where(labels == ci)[0]
                cluster_affinity = A[np.ix_(idx, idx)]
                rep = idx[np.argmax(cluster_affinity.sum(axis=1))]
                reps.append(rep)
            results["SpectralClusterReps"] = np.array(reps)
        except Exception:
            results["SpectralClusterReps"] = None

    return results
def select_topk_bands_spectral(A, n_clusters=10):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for spectral clustering")
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    band_labels = sc.fit_predict(A)
    top_k_bands = []
    for c in range(n_clusters):
        cluster_idx = np.where(band_labels == c)[0]
        cluster_affinity = A[np.ix_(cluster_idx, cluster_idx)]
        best_band = cluster_idx[np.argmax(cluster_affinity.sum(axis=1))]
        top_k_bands.append(int(best_band))
    return top_k_bands, band_labels

def plot_importance(importance, title="Band importance"):
    plt.figure(figsize=(10,3))
    plt.plot(importance, marker='o')
    plt.title(title)
    plt.xlabel("Band index")
    plt.ylabel("Normalized importance (sum=1)")
    plt.grid(alpha=0.3)
    plt.show()

def plot_clusters(band_labels, B):
    plt.figure(figsize=(12,3))
    plt.scatter(np.arange(B), np.zeros(B), c=band_labels, cmap="tab10", s=50)
    plt.title("Spectral Band Clusters")
    plt.xlabel("Band index")
    plt.yticks([])
    plt.show()

def test_selected_bands(X, Y ,top_k_bands):
    X_selected = X[:,sorted(top_k_bands)]
    X_train, X_test, Y_train, Y_test = split_XY(X_selected, Y, test_size=0.2)

    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_clf.fit(X_train, Y_train)
    Y_pred_rf = rf_clf.predict(X_test)

    OA_rf = accuracy_score(Y_test, Y_pred_rf)
    AA_rf = np.mean([accuracy_score(Y_test[Y_test==c], Y_pred_rf[Y_test==c]) for c in np.unique(Y_test)])
    kappa_rf = cohen_kappa_score(Y_test, Y_pred_rf)

    print(f"Random Forest results using top-{len(top_k_bands)} bands:")
    print(f"Overall Accuracy (OA): {OA_rf:.4f}")
    print(f"Average Accuracy (AA): {AA_rf:.4f}")
    print(f"Kappa coefficient: {kappa_rf:.4f}\n")

    svc_clf = SVC(kernel='rbf', C=1000, gamma='scale', class_weight='balanced')
    svc_clf.fit(X_train, Y_train)
    Y_pred_svc = svc_clf.predict(X_test)

    OA_svc = accuracy_score(Y_test, Y_pred_svc)
    AA_svc = np.mean([accuracy_score(Y_test[Y_test==c], Y_pred_svc[Y_test==c]) for c in np.unique(Y_test)])
    kappa_svc = cohen_kappa_score(Y_test, Y_pred_svc)

    print(f"SVC results using top-{len(top_k_bands)} bands:")
    print(f"Overall Accuracy (OA): {OA_svc:.4f}")
    print(f"Average Accuracy (AA): {AA_svc:.4f}")
    print(f"Kappa coefficient: {kappa_svc:.4f}")

# ----------------------- Visualization helpers -----------------------
def plot_epoch_history(history, savefile=None):
    # multiple subplots for losses
    epochs = len(history['train_mse'])
    x = np.arange(1, epochs+1)
    plt.figure(figsize=(12,8))

    plt.subplot(3,2,1)
    plt.plot(x, history['train_mse'], label='train'); plt.plot(x, history['val_mse'], label='val'); plt.title('MSE'); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(3,2,2)
    plt.plot(x, history['train_l1'], label='train'); plt.plot(x, history['val_l1'], label='val'); plt.title('L1'); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(3,2,3)
    plt.plot(x, history['train_tv'], label='train'); plt.plot(x, history['val_tv'], label='val'); plt.title('TV'); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(3,2,4)
    plt.plot(x, history['train_self'], label='train'); plt.plot(x, history['val_self'], label='val'); plt.title('Self-expression loss'); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(3,2,5)
    plt.plot(x, history['train_sam'], label='train'); plt.plot(x, history['val_sam'], label='val'); plt.title('SAM (approx)'); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(3,2,6)
    plt.plot(x, history['train_msssim'], label='train'); plt.plot(x, history['val_msssim'], label='val'); plt.title('MS-SSIM (1 - value)'); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=200)
    plt.show()


def compute_per_band_rmse_and_snr(cube, recon_cube, mask=None):
    # cube, recon_cube: (H,W,B)
    H, W, B = cube.shape
    orig = cube.reshape(-1, B)  # (H*W, B)
    recon = recon_cube.reshape(-1, B)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        orig = orig[mask_flat]
        recon = recon[mask_flat]

    noise = orig - recon
    mse_per_band = np.mean(noise**2, axis=0)
    rmse_per_band = np.sqrt(mse_per_band)

    # signal power estimate (variance of orig per band)
    signal_var = np.var(orig, axis=0)
    # avoid div by zero
    eps = 1e-12
    snr_linear = signal_var / (mse_per_band + eps)
    snr_db = 10.0 * np.log10(snr_linear + eps)
    return rmse_per_band, snr_db

def plot_rmse_snr(rmse, snr_db, importance=None, bands=None, savefile=None):
    B = len(rmse)
    x = np.arange(B)
    fig, ax1 = plt.subplots(figsize=(12,4))

    ax1.plot(x, rmse, label='RMSE', marker='o')
    ax1.set_xlabel('Band index')
    ax1.set_ylabel('RMSE')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, snr_db, label='SNR (dB)', marker='x', linestyle='--', alpha=0.8)
    ax2.set_ylabel('SNR (dB)')

    if importance is not None:
        # show importance as a translucent bar under the RMSE curve (scaled)
        imp_scaled = (importance - importance.min()) / (importance.max() - importance.min() + 1e-12)
        ax1.fill_between(x, 0, imp_scaled * np.nanmax(rmse), color='gray', alpha=0.15, label='importance (scaled)')

    # legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Per-band RMSE and SNR')
    fig.tight_layout()
    if savefile:
        fig.savefig(savefile, dpi=200)
    plt.show()


def show_high_noise_band_images(cube, recon_cube, rmse_per_band, top_n=6, save_dir=None):
    # show top_n bands with highest RMSE (noisy bands)
    idx = np.argsort(rmse_per_band)[-top_n:][::-1]
    H, W, B = cube.shape
    for i, b in enumerate(idx):
        orig = cube[:,:,b]
        recon = recon_cube[:,:,b]
        diff = (orig - recon)
        plt.figure(figsize=(12,3))
        plt.subplot(1,3,1); plt.title(f'Orig band {b}'); plt.imshow(orig, cmap='viridis'); plt.colorbar()
        plt.subplot(1,3,2); plt.title(f'Recon band {b}'); plt.imshow(recon, cmap='viridis'); plt.colorbar()
        plt.subplot(1,3,3); plt.title(f'Diff band {b}'); plt.imshow(diff, cmap='seismic'); plt.colorbar()
        plt.suptitle(f'High-noise band {b} (RMSE={rmse_per_band[b]:.4f})')
        if save_dir:
            plt.savefig(f"{save_dir}/band_{b:03d}_noise.png", dpi=200)
        plt.show()
lambda_msssim = 1e-3

# ----------------------- Utilities (SSIM approx, SAM) -----------------------
def _gaussian_kernel(window_size=11, sigma=1.5, channels=1, device='cpu', dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(1) @ g.unsqueeze(0)
    kernel = kernel_1d / kernel_1d.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel

def ssim_index(img1, img2, window_size=11, sigma=1.5, L=1.0, K=(0.01, 0.03), device=None):
    if device is None:
        device = img1.device
    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    channels = img1.shape[1]
    kernel = _gaussian_kernel(window_size=window_size, sigma=sigma, channels=channels, device=device, dtype=img1.dtype)
    padding = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=padding, groups=channels)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, kernel, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, kernel, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1*img2, kernel, padding=padding, groups=channels) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return ssim_map.mean()

def msssim_index(img1, img2, levels=3, window_size=11, sigma=1.5, L=1.0):
    vals = []
    a1, a2 = 0.8, 0.2
    weights = [a1] + [a2/(levels-1)]*(levels-1)
    cur1, cur2 = img1, img2
    for l in range(levels):
        s = ssim_index(cur1, cur2, window_size=window_size, sigma=sigma, L=L, device=cur1.device)
        vals.append(float(s))
        if l < levels-1:
            cur1 = F.avg_pool2d(cur1, kernel_size=2, stride=2, padding=0)
            cur2 = F.avg_pool2d(cur2, kernel_size=2, stride=2, padding=0)
    ms = sum(w*v for w,v in zip(weights, vals))
    return torch.tensor(ms, device=img1.device)
def sam_loss(recon_flat, target_flat, mode='cosine'):
    eps = 1e-8
    r = recon_flat
    t = target_flat
    dot = (r * t).sum(dim=1)
    rn = torch.norm(r, dim=1)
    tn = torch.norm(t, dim=1)
    cos = dot / (rn * tn + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)
    return ang.mean()



