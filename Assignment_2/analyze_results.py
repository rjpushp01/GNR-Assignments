"""
Aggregate Analysis Script
==========================
After all scenarios are run, this script:
  1. Aggregates all metrics.json files across models and scenarios
  2. Generates cross-model comparison plots (few-shot, corruption, layer depth)
  3. Generates the model efficiency table (params, MACs, FLOPs)
  4. Summarises all results into aggregate_results.csv

Usage:
    python analyze_results.py
    python analyze_results.py --results_root results/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.visualization import (
    plot_fewshot_results, plot_corruption_heatmap,
    plot_accuracy_vs_depth, plot_finetune_comparison, save_metrics_json,
)

MODELS = ["resnet50", "inception_v3", "densenet121"]
PALETTE = ["#4C72B0", "#DD8452", "#55A868"]


def load_metrics(results_root: str, model: str, *path_parts) -> dict:
    """Load metrics.json from a results sub-directory."""
    path = os.path.join(results_root, model, *path_parts, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def aggregate_efficiency(results_root: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        info_path = os.path.join(results_root, model, "scenario1_linear_probe", "model_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            rows.append({
                "model": model,
                "total_params_M": info.get("total_params", 0) / 1e6,
                "macs_str": info.get("macs_str", "N/A"),
                "flops_str": info.get("flops_str", "N/A"),
            })
    return pd.DataFrame(rows)


def aggregate_fewshot(results_root: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        r = {}
        for pct in [5, 20, 100]:
            m = load_metrics(results_root, model, "scenario3_few_shot", f"pct_{pct}")
            r[f"pct_{pct}"] = m.get("best_val_acc", float("nan"))
        acc_100 = r["pct_100"]
        acc_5 = r["pct_5"]
        delta = (acc_100 - acc_5) / acc_100 if acc_100 > 0 else float("nan")
        rows.append({"model": model, **r, "relative_drop": delta})
    return pd.DataFrame(rows)


def aggregate_corruption(results_root: str) -> pd.DataFrame:
    corruptions = [
        "gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
        "motion_blur", "brightness_dark", "brightness_bright",
    ]
    rows = []
    for model in MODELS:
        clean_m = load_metrics(results_root, model, "scenario4_corruption", "gaussian_0.05")
        clean_acc = clean_m.get("clean_acc", float("nan"))
        row = {"model": model, "clean_acc": clean_acc}
        for c in corruptions:
            m = load_metrics(results_root, model, "scenario4_corruption", c)
            row[f"{c}_acc"] = m.get("corrupted_acc", float("nan"))
            row[f"{c}_ce"] = m.get("corruption_error", float("nan"))
            row[f"{c}_rr"] = m.get("relative_robustness", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_finetune(results_root: str) -> pd.DataFrame:
    strategies = ["linear_probe", "last_block", "full", "selective_20pct"]
    rows = []
    for model in MODELS:
        for s in strategies:
            m = load_metrics(results_root, model, "scenario2_finetuning", s)
            rows.append({
                "model": model,
                "strategy": s,
                "best_val_acc": m.get("best_val_acc", float("nan")),
                "trainable_pct": m.get("trainable_pct", float("nan")),
            })
    return pd.DataFrame(rows)


def aggregate_layer_probe(results_root: str) -> pd.DataFrame:
    depths = ["early", "mid", "final"]
    rows = []
    for model in MODELS:
        for d in depths:
            m = load_metrics(results_root, model, "scenario5_layer_probe", d)
            rows.append({
                "model": model,
                "depth": d,
                "layer_name": m.get("layer_name", "N/A"),
                "best_val_acc": m.get("best_val_acc", float("nan")),
                "feature_dim": m.get("feature_dim", float("nan")),
                "mean_norm": m.get("feature_norms", {}).get("mean_norm", float("nan")),
            })
    return pd.DataFrame(rows)


def plot_cross_model_fewshot(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    pcts = [5, 20, 100]
    x = np.arange(len(pcts))
    width = 0.25
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row.get(f"pct_{p}", 0) for p in pcts]
        ax.bar(x + i * width, vals, width, label=row["model"], color=PALETTE[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["5%", "20%", "100%"])
    ax.set_xlabel("Training Data %")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Few-Shot Performance — All Models")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "fewshot_comparison.png"))
    plt.close(fig)


def plot_cross_model_layer_probe(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model].sort_values("depth", key=lambda x: x.map({"early": 0, "mid": 1, "final": 2}))
        ax.plot(sub["depth"].values, sub["best_val_acc"].values,
                marker="o", linewidth=2, label=model, color=PALETTE[i])
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Layer Probe Accuracy vs Depth — All Models")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "layer_probe_comparison.png"))
    plt.close(fig)


def plot_cross_model_finetune(df: pd.DataFrame, save_dir: str):
    strategy_order = ["linear_probe", "last_block", "selective_20pct", "full"]
    df["strategy_order"] = df["strategy"].apply(lambda x: strategy_order.index(x) if x in strategy_order else 99)
    df = df.sort_values("strategy_order")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model]
        ax.plot(sub["trainable_pct"].values, sub["best_val_acc"].values,
                marker="o", linewidth=2, label=model, color=PALETTE[i])
        for _, row in sub.iterrows():
            ax.annotate(row["strategy"][:7], (row["trainable_pct"], row["best_val_acc"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.set_xlabel("% Trainable Parameters")
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title("Fine-Tuning Strategy Comparison — All Models")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "finetune_comparison.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate all scenario results")
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    rr = args.results_root
    agg_dir = os.path.join(rr, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    print("\n=== Aggregating Results ===\n")

    # Efficiency table
    df_eff = aggregate_efficiency(rr)
    if not df_eff.empty:
        df_eff.to_csv(os.path.join(agg_dir, "model_efficiency.csv"), index=False)
        print("Model Efficiency:")
        print(df_eff.to_string(index=False))

    # Few-shot
    df_fs = aggregate_fewshot(rr)
    df_fs.to_csv(os.path.join(agg_dir, "fewshot_results.csv"), index=False)
    plot_cross_model_fewshot(df_fs, agg_dir)
    print("\nFew-Shot Results:")
    print(df_fs.to_string(index=False))

    # Fine-tuning
    df_ft = aggregate_finetune(rr)
    df_ft.to_csv(os.path.join(agg_dir, "finetune_results.csv"), index=False)
    plot_cross_model_finetune(df_ft, agg_dir)

    # Corruption
    df_cr = aggregate_corruption(rr)
    df_cr.to_csv(os.path.join(agg_dir, "corruption_results.csv"), index=False)
    print("\nCorruption Results:")
    print(df_cr.head().to_string(index=False))

    # Layer probe
    df_lp = aggregate_layer_probe(rr)
    df_lp.to_csv(os.path.join(agg_dir, "layer_probe_results.csv"), index=False)
    plot_cross_model_layer_probe(df_lp, agg_dir)
    print("\nLayer Probe Results:")
    print(df_lp.to_string(index=False))

    print(f"\n✓ Aggregate results saved to: {agg_dir}")


if __name__ == "__main__":
    main()
