import argparse
import math
import os
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoTokenizer

from bayesian_laws_icl import llm


def flip_str(p: float) -> str:
    """Consistent string encoding for flip_prob in filenames."""
    return f"flip{p:.2f}".replace(".", "p")


def save_outputs(df: pd.DataFrame, model_name: str, dataset: str, flip_prob: float) -> None:
    """Write raw and mean CSVs for a given run."""
    os.makedirs("logs/real-lms", exist_ok=True)
    m = model_name.replace("/", "_")
    d = dataset.replace("/", "_")
    suffix = flip_str(flip_prob)
    mean_path = os.path.join("logs/real-lms", f"{m}___{d}___{suffix}___means.csv")
    raw_path = os.path.join("logs/real-lms", f"{m}___{d}___{suffix}___raw.csv")

    df["prob"] = df["nll"].map(lambda x: math.exp(-x))
    df["shots"] += 1

    grouped = df.groupby(
        ["dataset", "shots", "model", "hmm", "flip_prob"], as_index=False
    ).mean(numeric_only=True)
    grouped["nll_avg"] = grouped["nll"]
    grouped["nll"] = grouped["prob"].map(lambda x: -math.log(x))

    grouped.to_csv(mean_path, index=False)
    df.to_csv(raw_path, index=False)
    print(f"Saved: {mean_path}")


def run_single(
    model,
    tokenizer,
    model_name: str,
    dataset: str,
    ct: int,
    shots: int,
    flip_prob: float,
    seed: int,
    use_together: bool,
    force: bool,
) -> None:
    """Run a single dataset/flip_prob combo and persist results."""
    suffix = flip_str(flip_prob)
    out_path = os.path.join(
        "logs/real-lms",
        f"{model_name.replace('/', '_')}___{dataset.replace('/', '_')}___{suffix}___means.csv",
    )
    if os.path.exists(out_path) and not force:
        print(f"Skip existing: {out_path}")
        return

    if use_together:
        data = llm.test_model_together(
            model_name,
            tokenizer,
            evals=[dataset],
            ct=ct,
            shots=shots,
            flip_prob=flip_prob,
            seed=seed,
        )
    else:
        data = llm.test_model(
            model,
            tokenizer,
            evals=[dataset],
            ct=ct,
            shots=shots,
            flip_prob=flip_prob,
            seed=seed,
        )
    df = pd.DataFrame(data)
    if df.empty:
        print(f"No data produced for {dataset} (flip_prob={flip_prob}); skipping save.")
        return
    save_outputs(df, model_name=model_name, dataset=dataset, flip_prob=flip_prob)
    torch.cuda.empty_cache()


def main(
    model_name: str,
    datasets: Iterable[str] | None,
    flip_probs: list[float],
    ct: int,
    shots: int,
    seed: int,
    force: bool,
) -> None:
    evals = list(datasets) if datasets else list(llm.dataset_info.keys())

    if model_name in llm.model_info:
        model, tokenizer = llm.load_model(model_name)
        use_together = False
    elif model_name in llm.model_info_together:
        tokenizer = AutoTokenizer.from_pretrained(
            llm.model_info_together[model_name]["tokenizer"]
        )
        model = None
        use_together = True
    else:
        raise ValueError(f"model {model_name} not found in configs")

    print(f"Running ablation for flip_probs={flip_probs}")
    for p in flip_probs:
        for dataset in evals:
            print(f"\n==> model={model_name} dataset={dataset} flip_prob={p}")
            run_single(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                dataset=dataset,
                ct=ct,
                shots=shots,
                flip_prob=p,
                seed=seed,
                use_together=use_together,
                force=force,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run flip_prob ablations for llm.py.")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b-it",
        help="HF model name or key in model_info / model_info_together.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets to run; if omitted, uses all from llm.dataset_info.",
    )
    parser.add_argument(
        "--flip_probs",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        help="List of flip probabilities to evaluate.",
    )
    parser.add_argument(
        "--ct",
        type=int,
        default=50,
        help="Number of random shuffles / prompts per dataset.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Maximum number of in-context demonstrations per prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for RNGs controlling flips and shuffling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output CSVs already exist.",
    )
    args = parser.parse_args()

    main(
        model_name=args.model,
        datasets=args.datasets,
        flip_probs=args.flip_probs,
        ct=args.ct,
        shots=args.shots,
        seed=args.seed,
        force=args.force,
    )
