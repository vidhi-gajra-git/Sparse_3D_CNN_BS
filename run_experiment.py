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
# ----------Helper functions to be shifted to utils ------------------------------
import torch
def cast_to_float(d, keys):
    for k in keys:
        if k in d:
            d[k] = float(d[k])


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
def save_latex_table(avg_df, out_path):
    """
    Saves a LaTeX table with mean ± std metrics.
    """
    df = avg_df.copy()

    df["OA"] = df["OA_mean"].map("{:.4f}".format) + " ± " + df["OA_std"].map("{:.4f}".format)
    df["AA"] = df["AA_mean"].map("{:.4f}".format) + " ± " + df["AA_std"].map("{:.4f}".format)
    df["Kappa"] = df["Kappa_mean"].map("{:.4f}".format) + " ± " + df["Kappa_std"].map("{:.4f}".format)

    latex_df = df[["classifier", "num_bands", "OA", "AA", "Kappa"]]
    latex_df = latex_df.sort_values(["classifier", "num_bands"])

    latex_str = latex_df.to_latex(
        index=False,
        escape=False,
        caption="Classification performance for different band counts (mean ± std over runs).",
        label="tab:band_selection_results"
    )

    with open(out_path, "w") as f:
        f.write(latex_str)


def plot_combined_accuracy(avg_df, out_dir, metric="OA_mean"):
    """
    Combined plot: accuracy vs bands for all classifiers.
    """
    plt.figure(figsize=(7, 5))

    for clf in avg_df["classifier"].unique():
        sub = avg_df[avg_df["classifier"] == clf].sort_values("num_bands")

        x = sub["num_bands"].values
        y = sub[metric].values

        best_idx = y.argmax()

        plt.plot(x, y, marker="o", linewidth=2, label=clf.upper())
        plt.scatter(
            x[best_idx],
            y[best_idx],
            s=80,
            zorder=3
        )

    plt.xlabel("Number of selected bands")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Performance vs Band Count ({metric.replace('_', ' ')})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"combined_{metric}_vs_bands.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

# -------------------------------------------------------------------------------------
def run_from_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    N_RUNS = cfg["experiment"]["runs"]

    dataset = cfg["data"]["dataset"]
    mat_path = cfg["data"]["mat_path"]
    gt_path = cfg["data"]["gt_path"]

    # ---- output dir ----
    out_dir = f"{dataset}/runs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load data ----
    cube, X, Y = load_hsi(mat_path, gt_path)
    H, W, B = cube.shape

    results = []
    band_metrics_all = []

    # ---- hyperparam search (optional) ----
    if cfg["experiment"]["hyperparam_search"]:
        search_df, best_params = run_hyperparam_search(cfg, cube, X, Y, device)
        search_df.to_csv(f"{out_dir}/hyperparam_search.csv", index=False)
        cfg["model"].update(best_params)

    # ---- cast yaml floats ----
    cast_to_float(cfg["training"], ["lr", "min_delta"])
    cast_to_float(
        cfg["regularization"],
        [
            "alpha", "beta", "lambda_l1", "lambda_tv",
            "C_init_scale", "C_lr_mult",
            "lambda_msssim", "lambda_C_l1"
        ]
    )

    # ================= MAIN RUN LOOP =================
    for run_id in range(N_RUNS):
        start_time = time.time()

        model = HybridModel(**cfg["model"], H=H, W=W).to(device)

        model, Z, C, A, history, recon = train_model(
            model=model,
            hsi=cube,
            device=device,
            **cfg["training"],
            **cfg["regularization"],
        )

        plot_epoch_history(
            history,
            savefile=f"{out_dir}/{run_id}_training_history.png"
        )

        rmse_per_band, snr_db = compute_per_band_rmse_and_snr(cube, recon)
        band_imp = get_band_importance_from_dict(A, method="l1")

        plot_rmse_snr(
            rmse_per_band,
            snr_db,
            importance=band_imp,
            savefile=f"{out_dir}/{run_id}_rmse_snr.png"
        )

        band_metrics_all.append(band_imp)

        train_time = time.time() - start_time
        size_mb = model_size_mb(model)

        clf_results = evaluate_classifiers(
            X=X,
            Y=Y,
            band_imp=band_imp,
            band_sizes=cfg["band_selection"]["topk"],
            classifiers=cfg["classifiers"]
        )

        for r in clf_results:
            r.update({
                "experiment": exp_name,
                "dataset": dataset,
                "run": run_id,
                "train_time_sec": train_time,
                "model_size_mb": size_mb
            })
            results.append(r)

    # ================= SAVE CSVs =================
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir}/results_per_run.csv", index=False)

    avg_df = (
        results_df
        .groupby(["classifier", "num_bands"])
        .agg(
            OA_mean=("OA", "mean"),
            OA_std=("OA", "std"),
            AA_mean=("AA", "mean"),
            AA_std=("AA", "std"),
            Kappa_mean=("Kappa", "mean"),
            Kappa_std=("Kappa", "std"),
            train_time_mean=("train_time_sec", "mean"),
            model_size_mb_mean=("model_size_mb", "mean")
        )
        .reset_index()
    )

    avg_df["experiment"] = exp_name
    avg_df["dataset"] = dataset
    avg_df["num_runs"] = N_RUNS

    avg_df.to_csv(f"{out_dir}/results_avg.csv", index=False)
    plot_combined_accuracy(avg_df, out_dir, metric="OA_mean")
    plot_combined_accuracy(avg_df, out_dir, metric="AA_mean")
    plot_combined_accuracy(avg_df, out_dir, metric="Kappa_mean")
    save_latex_table(avg_df, f"{out_dir}/results_table.tex")



    band_imp_df = pd.DataFrame(band_metrics_all)
    band_imp_df.to_csv(f"{out_dir}/band_metrics_per_run.csv", index=False)

    band_imp_df.mean(axis=0).to_frame("mean_importance").to_csv(
        f"{out_dir}/band_metrics_avg.csv"
    )

    print(f"✅ Finished experiment {exp_name} on {dataset}")
if __name__ == "__main__":
    config_files = [
        "configs/indian_pines.yaml",
        "configs/salinas.yaml",
        "configs/pavia.yaml",
        "configs/ksc.yaml"
    ]

    for cfg_path in config_files:
        run_from_config(cfg_path)



# # ---------------- Load config ----------------
# import torch
# def cast_to_float(d, keys):
#     for k in keys:
#         if k in d:
#             d[k] = float(d[k])


# def model_size_mb(model):
#     """
#     Returns the size of a PyTorch model in megabytes (MB),
#     including parameters and buffers.
#     """
#     param_bytes = 0
#     buffer_bytes = 0

#     for p in model.parameters():
#         param_bytes += p.numel() * p.element_size()

#     for b in model.buffers():
#         buffer_bytes += b.numel() * b.element_size()

#     total_bytes = param_bytes + buffer_bytes
#     size_mb = total_bytes / (1024 ** 2)

#     return round(size_mb, 3)

# with open("configs/exp1.yaml") as f:
#     cfg = yaml.safe_load(f)

# exp_name = cfg["experiment"]["name"]
# N_RUNS = cfg["experiment"]["runs"]

# os.makedirs(f"runs/{exp_name}/images", exist_ok=True)
# mat_path = cfg["data"]["mat_path"]
# gt_path = cfg["data"]["gt_path"]
# data_name=cfg["data"]["dataset"]
# # ---------------- Load data ----------------
# cube, X, Y = load_hsi(mat_path, gt_path)

# results = []
# band_metrics_all = []
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ---------------- Optional hyperparameter search ----------------
# if cfg["experiment"]["hyperparam_search"]:
#     search_df, best_params = run_hyperparam_search(cfg,cube,X,Y,device)
#     search_df.to_csv(f"{data_name}/runs/{exp_name}/hyperparam_search.csv", index=False)
#     cfg["model"].update(best_params)
# # -------------------Creating the folders for saving the data---------------------

# out_dir = f"{cfg['data']['dataset']}/runs/{cfg['experiment']['name']}"
# os.makedirs(out_dir, exist_ok=True)

# # ---------------- Main loop ----------------
# for run_id in range(N_RUNS):
    
#     start_time = time.time()
#     H,W,B=cube.shape

#     model = HybridModel(**cfg["model"],H=H,W=W)
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     cast_to_float(
#     cfg["training"],
#     ["lr", "min_delta"]
#         )

#     cast_to_float(
#         cfg["regularization"],
#         [
#             "alpha",
#             "beta",
#             "lambda_l1",
#             "lambda_tv",
#             "C_init_scale",
#             "C_lr_mult",
#             "lambda_msssim",
#             "lambda_C_l1"
#         ]
#     )

#     model, Z, C, A, history, recon = train_model(
#         model=model,
#         hsi=cube,
#         device=device,
#         **cfg["training"],
#         **cfg["regularization"],
       
#     )

    
#     plot_epoch_history(history, savefile=f'{data_name}/runs/{exp_name}/{run_id}_training_history.png')

# # compute per-band RMSE and SNR (use all pixels mask)
#     rmse_per_band, snr_db = compute_per_band_rmse_and_snr(cube, recon, mask=None)
    

#     train_time = time.time() - start_time
#     size_mb = model_size_mb(model)

#     # ---------- Band importance ----------
#     band_imp = get_band_importance_from_dict(A, method='l1')
#     band_metrics_all.append(band_imp)
#     plot_rmse_snr(rmse_per_band, snr_db, importance=band_imp, savefile=f'{data_name}/runs/{exp_name}/{run_id}_rmse_snr.png')

#     # ---------- Classifier evaluation ----------
#     clf_results = evaluate_classifiers(
#     X=X,
#     Y=Y,
#     band_imp=band_imp,
#     band_sizes=(20, 25, 30),
#     classifiers=cfg["classifiers"]
# )

# for r in clf_results:
#     r.update({
#         "experiment": exp_name,
#         "run": run_id,
#         "train_time_sec": train_time,
#         "model_size_mb": size_mb
#     })
#     results.append(r)

# # ---------------- Save per-run results ----------------
# results_df = pd.DataFrame(results)
# results_df.to_csv(
#     f"{data_name}/runs/{exp_name}/results_per_run.csv",
#     index=False
# )

# # ---------------- Compute averages across runs ----------------
# avg_df = (
#     results_df
#     .groupby(["classifier", "num_bands"])
#     .agg(
#         OA_mean=("OA", "mean"),
#         OA_std=("OA", "std"),
#         AA_mean=("AA", "mean"),
#         AA_std=("AA", "std"),
#         Kappa_mean=("Kappa", "mean"),
#         Kappa_std=("Kappa", "std"),
#         train_time_mean=("train_time_sec", "mean"),
#         model_size_mb_mean=("model_size_mb", "mean")
#     )
#     .reset_index()
# )

# avg_df["experiment"] = exp_name
# avg_df["num_runs"] = N_RUNS

# avg_df.to_csv(
#     f"{data_name}/runs/{exp_name}/results_avg.csv",
#     index=False
# )
# band_imp_df = pd.DataFrame(band_metrics_all)
# band_imp_df.to_csv(
#     f"{data_name}/runs/{exp_name}/band_metrics_per_run.csv",
#     index=False
# )

# band_imp_mean = band_imp_df.mean(axis=0)
# band_imp_mean.to_frame("mean_importance").to_csv(
#     f"{data_name}/runs/{exp_name}/band_metrics_avg.csv"
# )


