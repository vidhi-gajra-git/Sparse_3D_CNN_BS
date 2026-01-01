import yaml, time, json, os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data import load_hsi
from src.model import HybridModel
from src.train import train_model
from src.utils import plot_epoch_history, compute_per_band_rmse_and_snr, plot_rmse_snr, get_band_importance_from_dict
from src.classifiers import evaluate_classifiers, plot_band_importance
from src.search_param import run_hyperparam_search
# from src.utils import save_training_plots
# ----------Helper functions to be shifted to utils ------------------------------
import torch
from scipy.stats import wilcoxon
from scipy.stats import kendalltau
# ================== GLOBAL SEED SETUP ==================
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism flags (important)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For CUDA >= 10.2 (extra safety)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"🌱 Global seed set to {seed}")



def band_ranking_agreement(band_imp_df):
    """
    Computes average Kendall Tau across all pairs of runs.
    """
    rankings = [
        np.argsort(-band_imp_df.iloc[i].values)
        for i in range(len(band_imp_df))
    ]

    taus = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            tau, _ = kendalltau(rankings[i], rankings[j])
            taus.append(tau)

    return np.mean(taus), np.std(taus)


def significance_test(results_df, classifier, bands_a, bands_b):
    a = results_df[
        (results_df["classifier"] == classifier) &
        (results_df["num_bands"] == bands_a)
    ]["OA"].values

    b = results_df[
        (results_df["classifier"] == classifier) &
        (results_df["num_bands"] == bands_b)
    ]["OA"].values

    stat, p = wilcoxon(a, b)
    return p

def save_hyperparams_txt(cfg, out_dir):
    with open(f"{out_dir}/hyperparams_used.txt", "w") as f:
        f.write("# Training\n")
        f.write(yaml.dump(cfg["training"], sort_keys=False))
        f.write("\n# Model\n")
        f.write(yaml.dump(cfg["model"], sort_keys=False))
        f.write("\n# Regularization\n")
        f.write(yaml.dump(cfg["regularization"], sort_keys=False))

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
import seaborn as sns
import numpy as np

def plot_band_importance_ci(band_imp_df, out_dir):
    """
    Plots mean ± 95% CI for band importance across runs.
    """
    df = band_imp_df.copy()
    df["run"] = np.arange(len(df))
    long_df = df.melt(id_vars="run", var_name="band", value_name="importance")

    plt.figure(figsize=(10, 4))
    sns.regplot(
        data=long_df,
        x="band",
        y="importance",
        scatter=False,
        ci=95,
        line_kws={"linewidth": 2}
    )

    plt.title("Band importance with 95% confidence interval")
    plt.xlabel("Band index")
    plt.ylabel("Importance")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{out_dir}/band_importance_ci.png", dpi=200)
    plt.close()



def plot_combined_metrics(avg_df, out_dir):
    """
    For each classifier:
    - One figure
    - OA, AA, Kappa plotted together
    - Different line styles
    - Best point marked for each metric
    """
    metrics = {
        "OA_mean": {"label": "OA", "linestyle": "-", "marker": "o"},
        "AA_mean": {"label": "AA", "linestyle": "--", "marker": "s"},
        "Kappa_mean": {"label": "Kappa", "linestyle": ":", "marker": "^"},
    }

    for clf in avg_df["classifier"].unique():
        sub = avg_df[avg_df["classifier"] == clf].sort_values("num_bands")
        x = sub["num_bands"].values

        plt.figure(figsize=(7, 5))

        for metric, style in metrics.items():
            y = sub[metric].values
            best_idx = y.argmax()

            plt.plot(
                x, y,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                label=style["label"]
            )

            plt.scatter(
                x[best_idx],
                y[best_idx],
                s=100,
                zorder=3
            )

        plt.xlabel("Number of selected bands")
        plt.ylabel("Performance")
        plt.title(f"{clf.upper()} – Performance vs Band Count")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"{out_dir}/{clf}_all_metrics_vs_bands.png",
            dpi=200
        )
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
    # BASE_SEED = cfg["experiment"].get("seed", 42)

    # ================= MAIN RUN LOOP =================
    for run_id in range(N_RUNS):
        # for run_id in range(N_RUNS):
        # set_global_seed(BASE_SEED + run_id)
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
        plot_band_importance(band_imp,outdir=f"{out_dir}/{run_id}_band_imp.png")

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
    plot_combined_metrics(avg_df, out_dir)
    
    save_latex_table(avg_df, f"{out_dir}/results_table.tex")



    band_imp_df = pd.DataFrame(band_metrics_all)
    band_imp_df.to_csv(f"{out_dir}/band_metrics_per_run.csv", index=False)

    band_imp_df.mean(axis=0).to_frame("mean_importance").to_csv(
        f"{out_dir}/band_metrics_avg.csv"
    )
    print("🦋 Kendall tau values for the bands are (mean,std)", band_ranking_agreement(band_imp_df))
    p_val = significance_test(results_df, "svc", 20, 30)
    print(f"SVC: 20 vs 30 bands, p = {p_val:.4f}")
    p_val = significance_test(results_df, "rf", 20, 30)
    print(f"RF: 20 vs 30 bands, p = {p_val:.4f}")
    p_val = significance_test(results_df, "knn", 20, 30)
    print(f"KNN: 20 vs 30 bands, p = {p_val:.4f}")
    # plot_band_importance_ci(band_imp_df, out_dir)

    
    
    save_hyperparams_txt(cfg, out_dir)
    # ================= Accuracy using AVERAGED band importance =================
    mean_band_imp = band_imp_df.mean(axis=0).values
    
    avg_imp_results = evaluate_classifiers(
        X=X,
        Y=Y,
        band_imp=mean_band_imp,
        band_sizes=cfg["band_selection"]["topk"],
        classifiers=cfg["classifiers"]
    )
    
    for r in avg_imp_results:
        r.update({
            "experiment": exp_name,
            "dataset": dataset,
            "run": "avg_band_imp"
        })
    
    pd.DataFrame(avg_imp_results).to_csv(
        f"{out_dir}/results_avg_bandimp.csv",
        index=False
    )
    



    print(f"✅ Finished experiment {exp_name} on {dataset}")
if __name__ == "__main__":
    config_files = [
        "configs/indian_pines.yaml",
        "configs/salinas.yaml",
        "configs/pavia_u.yaml",
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


