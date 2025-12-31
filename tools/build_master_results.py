import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= USER CONFIG =================
RUNS_ROOT = "."   # root where dataset folders exist
METRICS = ["OA", "AA", "Kappa"]
CONF_Z = 1.96     # 95% confidence interval
# ==============================================


def load_all_results(runs_root):
    """
    Loads results_avg.csv from all datasets / experiments.
    """
    records = []

    for dataset in os.listdir(runs_root):
        dataset_dir = os.path.join(runs_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        runs_dir = os.path.join(dataset_dir, "runs")
        if not os.path.isdir(runs_dir):
            continue

        for exp in os.listdir(runs_dir):
            exp_dir = os.path.join(runs_dir, exp)
            avg_path = os.path.join(exp_dir, "results_avg.csv")

            if os.path.exists(avg_path):
                df = pd.read_csv(avg_path)
                df["dataset"] = dataset
                df["experiment"] = exp
                records.append(df)

    return pd.concat(records, ignore_index=True)


# ------------------------------------------------
def compute_confidence_interval(mean, std, n):
    return CONF_Z * (std / np.sqrt(n))


# ------------------------------------------------
def plot_confidence_intervals(master_df, out_dir, metric):
    """
    Plot mean ± confidence interval for all datasets & classifiers.
    """
    plt.figure(figsize=(8, 5))

    for clf in master_df["classifier"].unique():
        sub = master_df[master_df["classifier"] == clf]

        x = np.arange(len(sub))
        y = sub[f"{metric}_mean"].values
        ci = sub[f"{metric}_ci"].values

        plt.errorbar(
            x,
            y,
            yerr=ci,
            marker="o",
            linestyle="-",
            capsize=4,
            label=clf.upper()
        )

    plt.xticks(
        np.arange(len(sub)),
        sub["dataset"] + " (" + sub["num_bands"].astype(str) + ")",
        rotation=45,
        ha="right"
    )
    plt.ylabel(metric)
    plt.title(f"{metric} with 95% Confidence Interval")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{metric}_confidence_intervals.png"), dpi=200)
    plt.close()


# ------------------------------------------------
def make_latex_master_table(master_df, out_path):
    """
    Creates LaTeX table with bold best values per dataset & classifier.
    """
    df = master_df.copy()

    for metric in METRICS:
        col = f"{metric}_mean"
        for (dataset, clf), grp in df.groupby(["dataset", "classifier"]):
            best_idx = grp[col].idxmax()
            df.loc[best_idx, col] = (
                "\\textbf{" + f"{df.loc[best_idx, col]:.4f}" + "}"
            )

    # format strings
    for metric in METRICS:
        df[metric] = (
            df[f"{metric}_mean"].astype(str)
            + " ± "
            + df[f"{metric}_ci"].map(lambda x: f"{x:.4f}")
        )

    latex_df = df[
        ["dataset", "classifier", "num_bands", "OA", "AA", "Kappa"]
    ].sort_values(["dataset", "classifier", "num_bands"])

    latex = latex_df.to_latex(
        index=False,
        escape=False,
        caption="Master results across datasets (mean ± 95\\% CI). Best values in bold.",
        label="tab:master_results"
    )

    with open(out_path, "w") as f:
        f.write(latex)


# ====================== MAIN ======================
if __name__ == "__main__":
    master = load_all_results(RUNS_ROOT)

    # compute CI
    n_runs = master["num_runs"].iloc[0]
    for metric in METRICS:
        master[f"{metric}_ci"] = compute_confidence_interval(
            master[f"{metric}_mean"],
            master[f"{metric}_std"],
            n_runs
        )

    # save master CSV
    os.makedirs("master_results", exist_ok=True)
    master.to_csv("master_results/master_results.csv", index=False)

    # plots
    for metric in METRICS:
        plot_confidence_intervals(master, "master_results", metric)

    # LaTeX
    make_latex_master_table(
        master,
        "master_results/master_results.tex"
    )

    print("✅ Master table, CI plots, and LaTeX generated.")
