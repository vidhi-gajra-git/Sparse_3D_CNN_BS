from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import torch
import torch.nn as nn
import scipy.io
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
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
# ----------------------- Data utilities -----------------------
class BandPatchDataset(Dataset):
    """Generate per-band spectral patches for 3D CNN"""
    def __init__(self, hsi: np.ndarray, window_size:int=9, transform=None):
        assert hsi.ndim == 3, "HSI must be H x W x B"
        H, W, B = hsi.shape
        self.hsi = hsi.astype(np.float32)
        self.H = H; self.W = W; self.B = B
        self.k = window_size // 2
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        b = int(idx)
        left = max(0, b - self.k)
        right = min(self.B - 1, b + self.k)
        indices = list(range(left, right + 1))
        # pad by repeating edge bands if necessary to reach window_size
        while len(indices) < self.window_size:
            if indices[0] == 0:
                indices.insert(0,0)
            else:
                indices.insert(0, indices[0]-1)
        while len(indices) < self.window_size:
            if indices[-1] == self.B - 1:
                indices.append(self.B - 1)
            else:
                indices.append(indices[-1]+1)

        patch = self.hsi[:, :, indices]
        patch = np.transpose(patch, (2,0,1)).copy()   # (window_size, H, W)
        target = self.hsi[:, :, b].reshape(-1).copy() # vectorized target for reconstruction

        if self.transform is not None:
            patch = self.transform(patch)
            target = self.transform(target)

        patch = torch.from_numpy(patch)
        target = torch.from_numpy(target)
        return patch, target, b

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
