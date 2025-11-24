#!/usr/bin/env python
"""
fit_llm_flip.py

Fits Bayesian and non-Bayesian scaling laws to LLM ICL curves under label flips,
using the existing `compute_all_fits` function from analyse.py, and visualises:

- NRMSE vs flip_prob: Bayesian (scoring) vs best non-Bayesian baseline
- Bayesian efficiency K vs flip_prob

Expected input files (for a given model & dataset):

    logs/real-lms/<model>___<dataset>___flipXpYY___means.csv

where flipXpYY encodes flip_prob, e.g. flip0p25 -> 0.25

Example usage (from repo root):

    python fit_llm_flip.py \
        --model meta-llama_Llama-3.1-8B-Instruct \
        --dataset creak

This will read all matching means CSVs, run the fits, and save:

    logs/real-lms/<model>___<dataset>___flip_fits.csv
    logs/real-lms/<model>___<dataset>___nrmse_vs_flip.png
    logs/real-lms/<model>___<dataset>___bayesK_vs_flip.png
"""

import argparse
import glob
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Adjust this import depending on where you put the script:
# - If this file lives at repo root: from bayesian_laws_icl.analyse import compute_all_fits
# - If this file lives inside bayesian_laws_icl/: from .analyse import compute_all_fits
from bayesian_laws_icl.analyse import compute_all_fits


def parse_flip_prob_from_fname(fname: str) -> float:
    """
    Extract flip probability from a filename segment like '...___flip0p25___...'
    -> 0.25
    """
    m = re.search(r"___flip([0-9p]+)___", fname)
    if not m:
        raise ValueError(f"Could not parse flip_prob from filename: {fname}")
    flip_str = m.group(1)  # e.g. '0p25'
    return float(flip_str.replace("p", "."))


def load_means_files(model: str, dataset: str, base_dir: str = "logs/real-lms") -> pd.DataFrame:
    """
    Load all *___means.csv files for a given model and dataset and return a
    single DataFrame with a flip_prob column.
    """
    pattern = os.path.join(
        base_dir, f"{model}___{dataset}___flip*___means.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No means CSVs found matching {pattern}")

    dfs = []
    for path in files:
        fname = os.path.basename(path)
        flip_prob = parse_flip_prob_from_fname(fname)
        df = pd.read_csv(path)
        df["flip_prob"] = flip_prob
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def fit_laws_vs_flip(creak_means: pd.DataFrame,
                     max_shots: float = 0.5,
                     epochs: int = 50,
                     patience: int = 5,
                     lr: float = 5e-2) -> pd.DataFrame:
    """
    For each flip_prob in the combined means DataFrame, run compute_all_fits on
    prob vs shots (loss_mode='mse_prob') and return a DataFrame with all fit
    results.

    We use:
      - mode='adam'
      - max_shots as a fraction of available shots (e.g. 0.5 for first half)
      - epochs & patience as given
    """
    results = []

    # group by flip_prob, fit all laws separately for each noise level
    for fp, df_fp in creak_means.groupby("flip_prob"):
        subset = df_fp.copy()
        num_hmms = subset["hmm"].nunique()

        # compute_all_fits returns (params_list, models)
        params_list, models = compute_all_fits(
            subset=subset,
            max_shots=max_shots,     # fraction of max shots (0.5 = first half)
            quiet=True,
            patience=patience,
            epochs=epochs,
            lr=lr,
            num_hmms=num_hmms,
            i=0,
            metadata={"flip_prob": fp},
            mode="adam",             # optimiser
            loss_mode="mse_prob",    # fit on probability vs shots
            log_shots=False,
            sweep=False,
        )

        for p in params_list:
            p["flip_prob"] = fp
        results.extend(params_list)

    res_df = pd.DataFrame(results)
    return res_df


def summarise_nrmse(res_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise NRMSE for Bayesian scoring vs best non-Bayesian baseline
    (power/bounded/logistic) as a function of flip_prob.

    Assumes res_df has columns: ['law', 'hmm', 'nrmse_prob', 'flip_prob', ...]
    and we care about hmm=0 (CREAK/LogiQA single-task case).
    """
    # Focus on hmm=0 and non-log-shots fits
    df = res_df[res_df["hmm"] == 0].copy()

    rows = []
    for fp, df_fp in df.groupby("flip_prob"):
        # Bayesian scoring law
        bayes_row = df_fp[df_fp["law"] == "bayesian_scoring"]
        if bayes_row.empty:
            raise ValueError(
                f"No bayesian_scoring law found for flip_prob={fp}")
        bayes_nrmse = float(bayes_row["nrmse_prob"].iloc[0])

        # Best non-Bayesian among power/bounded/logistic
        non_bayes_df = df_fp[df_fp["law"].isin(
            ["power", "bounded", "logistic"])]
        best_non_nrmse = float(non_bayes_df["nrmse_prob"].min())
        best_non_law = str(
            non_bayes_df.loc[non_bayes_df["nrmse_prob"].idxmin(), "law"]
        )

        rows.append(
            {
                "flip_prob": fp,
                "bayes_nrmse": bayes_nrmse,
                "best_non_nrmse": best_non_nrmse,
                "best_non_law": best_non_law,
            }
        )

    summary = pd.DataFrame(rows).sort_values("flip_prob")
    return summary


def plot_nrmse_vs_flip(summary: pd.DataFrame, out_path: str):
    """
    Plot Bayesian vs best non-Bayesian NRMSE vs flip_prob and save to out_path.
    """
    plt.figure()
    plt.plot(
        summary["flip_prob"], summary["bayes_nrmse"],
        marker="o", label="Bayesian (scoring)"
    )
    plt.plot(
        summary["flip_prob"], summary["best_non_nrmse"],
        marker="o", label="Best non-Bayesian"
    )
    plt.xlabel("Flip probability")
    plt.ylabel("NRMSE (prob)")
    plt.title("NRMSE vs label flip probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_bayesK_vs_flip(res_df: pd.DataFrame, out_path: str):
    """
    Plot Bayesian efficiency K vs flip_prob for the bayesian_scoring law.
    """
    df = res_df[(res_df["law"] == "bayesian_scoring") & (res_df["hmm"] == 0)]
    bayes_rows = (
        df[["flip_prob", "K"]]
        .drop_duplicates()
        .sort_values("flip_prob")
    )

    plt.figure()
    plt.plot(bayes_rows["flip_prob"], bayes_rows["K"], marker="o")
    plt.xlabel("Flip probability")
    plt.ylabel("Bayesian efficiency K")
    plt.title("Bayesian K vs label flip probability")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name as used in logs filenames, e.g. "
             "meta-llama_Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name as used in logs filenames, e.g. creak or logiqa",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="logs/real-lms",
        help="Base directory where *___means.csv files are stored.",
    )
    parser.add_argument(
        "--max_shots",
        type=float,
        default=0.5,
        help="If <=1, treat as fraction of max shots to use; "
             "if >1, treat as absolute max shots.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs for scaling-law fits.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience for scaling-law fits.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-2,
        help="Learning rate for scaling-law fits.",
    )
    args = parser.parse_args()

    # Load all means files for this model & dataset
    df_means = load_means_files(
        args.model, args.dataset, base_dir=args.base_dir)

    # Fit all laws per flip_prob
    res_df = fit_laws_vs_flip(
        df_means,
        max_shots=args.max_shots,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )

    # Prepare output paths
    out_dir = Path(args.base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fits_csv = out_dir / f"{args.model}___{args.dataset}___flip_fits.csv"
    nrmse_png = out_dir / f"{args.model}___{args.dataset}___nrmse_vs_flip.png"
    k_png = out_dir / f"{args.model}___{args.dataset}___bayesK_vs_flip.png"

    # Save full fit results
    res_df.to_csv(fits_csv, index=False)

    # Summarise NRMSE for Bayesian vs best non-Bayesian
    nrmse_summary = summarise_nrmse(res_df)
    print("NRMSE summary:")
    print(nrmse_summary)

    # Plot NRMSE vs flip_prob
    plot_nrmse_vs_flip(nrmse_summary, str(nrmse_png))

    # Plot Bayesian K vs flip_prob
    plot_bayesK_vs_flip(res_df, str(k_png))

    print(f"Saved fits to: {fits_csv}")
    print(f"Saved NRMSE plot to: {nrmse_png}")
    print(f"Saved Bayesian K plot to: {k_png}")


if __name__ == "__main__":
    main()
