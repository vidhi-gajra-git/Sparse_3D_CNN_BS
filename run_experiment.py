import yaml, time, json, os
import pandas as pd
import torch

from src.data import load_hsi
from src.model import HybridModel
from src.train import train_model
from src.utils import plot_epoch_history, compute_per_band_rmse_and_snr, plot_rmse_snr, get_band_importance_from_dict
from src.classifiers import evaluate_classifiers
from src.search_param import run_hyperparam_search
# from src.utils import save_training_plots

# ---------------- Load config ----------------
import torch

def model_size_mb(model):
    """
    Returns the size of a PyTorch model in megabytes (MB),
    including parameters and buffers.
    """
    param_bytes = 0
    buffer_bytes = 0

    for p in model.parameters():
        param_bytes += p.numel() * p.element_size()

    for b in model.buffers():
        buffer_bytes += b.numel() * b.element_size()

    total_bytes = param_bytes + buffer_bytes
    size_mb = total_bytes / (1024 ** 2)

    return round(size_mb, 3)

with open("configs/exp1.yaml") as f:
    cfg = yaml.safe_load(f)

exp_name = cfg["experiment"]["name"]
N_RUNS = cfg["experiment"]["runs"]

os.makedirs(f"runs/{exp_name}/images", exist_ok=True)
mat_path = cfg["data"]["mat_path"]
gt_path = cfg["data"]["gt_path"]
data_name=cfg["data"]["dataset"]
# ---------------- Load data ----------------
cube, X, Y = load_hsi(mat_path, gt_path)

results = []
band_metrics_all = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------- Optional hyperparameter search ----------------
if cfg["experiment"]["hyperparam_search"]:
    search_df, best_params = run_hyperparam_search(cfg,cube,X,Y,device)
    search_df.to_csv(f"{data_name}/runs/{exp_name}/hyperparam_search.csv", index=False)
    cfg["model"].update(best_params)

# ---------------- Main loop ----------------
for run_id in range(N_RUNS):
    
    start_time = time.time()
    H,W,B=cube.shape

    model = HybridModel(**cfg["model"],H=H,W=W)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model, Z, C, A, history, recon = train_model(
        model=model,
        cube=cube,
        device=device,
        **cfg["training"],
        **cfg["regularization"],
       
    )
    plot_epoch_history(history, savefile='{data_name}/runs/{exp_name}/{run_id}_training_history.png')

# compute per-band RMSE and SNR (use all pixels mask)
    rmse_per_band, snr_db = compute_per_band_rmse_and_snr(cube, recon, mask=None)
    plot_rmse_snr(rmse_per_band, snr_db, importance=band_imp, savefile='{data_name}/runs/{exp_name}/{run_id}_rmse_snr.png')

    train_time = time.time() - start_time
    size_mb = model_size_mb(model)

    # ---------- Band importance ----------
    band_imp = get_band_importance_from_dict(A, method='l1')
    band_metrics_all.append(band_imp)

    # ---------- Classifier evaluation ----------
    clf_results = evaluate_classifiers(
    X=X,
    Y=Y,
    ranked_bands=band_imp,
    band_sizes=(20, 25, 30),
    classifiers=cfg["classifiers"]
)

for r in clf_results:
    r.update({
        "experiment": exp_name,
        "run": run_id,
        "train_time_sec": train_time,
        "model_size_mb": size_mb
    })
    results.append(r)

# ---------------- Save CSVs ----------------
pd.DataFrame(results).to_csv(f"{data_name}/runs/{exp_name}/results.csv", index=False)
pd.DataFrame(band_metrics_all).to_csv(f"{data_name}/runs/{exp_name}/band_metrics.csv", index=False)
