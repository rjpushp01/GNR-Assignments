import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MODELS = ["resnet50", "inception_v3", "densenet121"]
PALETTE = ["#4C72B0", "#DD8452", "#55A868"]

def load_metrics(results_root: str, model: str, *path_parts) -> dict:
    path = os.path.join(results_root, model, *path_parts, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def load_history(results_root: str, model: str, *path_parts) -> dict:
    path = os.path.join(results_root, model, *path_parts, "history.json")
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
                "trainable_params_M": info.get("trainable_params", 0) / 1e6,
                "frozen_params_M": info.get("frozen_params", 0) / 1e6,
                "macs_str": info.get("macs_str", "N/A"),
                "flops_str": info.get("flops_str", "N/A"),
            })
    return pd.DataFrame(rows)

def aggregate_scenario1(results_root: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        m = load_metrics(results_root, model, "scenario1_linear_probe")
        if m:
            rows.append({
                "model": model,
                "best_val_acc": m.get("best_val_acc"),
                "total_training_time": m.get("total_training_time"),
                "train_pass_time": m.get("train_pass_time"),
                "val_pass_time": m.get("val_pass_time"),
                "data_fetch_time": m.get("data_fetch_time"),
                "dataset_setup_time": m.get("dataset_setup_time"),
                "avg_epoch_time": m.get("avg_epoch_time"),
            })
    return pd.DataFrame(rows)

def aggregate_finetune(results_root: str) -> pd.DataFrame:
    strategies = ["linear_probe", "last_block", "selective_20pct", "full"]
    rows = []
    for model in MODELS:
        for s in strategies:
            m = load_metrics(results_root, model, "scenario2_finetuning", s)
            if m:
                rows.append({
                    "model": model,
                    "strategy": s,
                    "best_val_acc": m.get("best_val_acc"),
                    "final_train_acc": m.get("final_train_acc"),
                    "final_val_acc": m.get("final_val_acc"),
                    "trainable_pct": m.get("trainable_pct"),
                    "total_training_time": m.get("total_training_time"),
                    "train_pass_time": m.get("train_pass_time"),
                    "val_pass_time": m.get("val_pass_time"),
                    "data_fetch_time": m.get("data_fetch_time"),
                    "dataset_setup_time": m.get("dataset_setup_time"),
                    "avg_epoch_time": m.get("avg_epoch_time"),
                })
    return pd.DataFrame(rows)

def aggregate_fewshot(results_root: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for pct in [5, 20, 100]:
            m = load_metrics(results_root, model, "scenario3_few_shot", f"pct_{pct}")
            if m:
                rows.append({
                    "model": model,
                    "pct": pct,
                    "best_val_acc": m.get("best_val_acc"),
                    "total_training_time": m.get("total_training_time"),
                    "train_pass_time": m.get("train_pass_time"),
                    "val_pass_time": m.get("val_pass_time"),
                    "data_fetch_time": m.get("data_fetch_time"),
                    "dataset_setup_time": m.get("dataset_setup_time"),
                    "avg_epoch_time": m.get("avg_epoch_time"),
                })
    return pd.DataFrame(rows)

def aggregate_corruption(results_root: str) -> pd.DataFrame:
    corruptions = [
        "gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
        "motion_blur", "brightness_dark", "brightness_bright",
    ]
    rows = []
    for model in MODELS:
        for c in corruptions:
            m = load_metrics(results_root, model, "scenario4_corruption", c)
            if m:
                rows.append({
                    "model": model,
                    "corruption": c,
                    "clean_acc": m.get("clean_acc"),
                    "corrupted_acc": m.get("corrupted_acc"),
                    "corruption_error": m.get("corruption_error"),
                    "relative_robustness": m.get("relative_robustness"),
                })
    return pd.DataFrame(rows)

def aggregate_layer_probe(results_root: str) -> pd.DataFrame:
    depths = ["early", "mid", "final"]
    rows = []
    for model in MODELS:
        for d in depths:
            m = load_metrics(results_root, model, "scenario5_layer_probe", d)
            if m:
                rows.append({
                    "model": model,
                    "depth": d,
                    "layer_name": m.get("layer_name"),
                    "best_val_acc": m.get("best_val_acc"),
                    "feature_dim": m.get("feature_dim"),
                    "mean_norm": m.get("feature_norms", {}).get("mean_norm"),
                    "extraction_time": m.get("extraction_time"),
                    "total_train_time": m.get("total_train_time"),
                    "avg_epoch_time": m.get("avg_epoch_time"),
                    "scenario_time_s": m.get("total_scenario_time"),
                })
    return pd.DataFrame(rows)

# =============== PLOTTING FUNCTIONS ===============

def do_stacked_time_plot(ax, df, x_col, x_labels, title):
    x = np.arange(len(x_labels))
    width = 0.25
    for i, model in enumerate(MODELS):
        model_df = df[df["model"] == model]
        if model_df.empty: continue
        
        y_train = []
        y_val = []
        y_fetch = []
        for val in x_labels:
            row = model_df[model_df[x_col] == val]
            if not row.empty:
                y_train.append(row["train_pass_time"].values[0])
                y_val.append(row["val_pass_time"].values[0])
                y_fetch.append(row["data_fetch_time"].values[0])
            else:
                y_train.append(0); y_val.append(0); y_fetch.append(0)
                
        bottom_val = np.array(y_train)
        bottom_fetch = bottom_val + np.array(y_val)
        
        ax.bar(x + (i - 1) * width, y_train, width, color=PALETTE[i], alpha=0.9)
        ax.bar(x + (i - 1) * width, y_val, width, bottom=bottom_val, color=PALETTE[i], alpha=0.6, hatch='//')
        ax.bar(x + (i - 1) * width, y_fetch, width, bottom=bottom_fetch, color=PALETTE[i], alpha=0.3, hatch='xx')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
        
    legend_elements = [
        Patch(facecolor=PALETTE[0], label='ResNet50'),
        Patch(facecolor=PALETTE[1], label='Inception_V3'),
        Patch(facecolor=PALETTE[2], label='DenseNet121'),
        Patch(facecolor='grey', alpha=0.9, label='Train Pass'),
        Patch(facecolor='grey', alpha=0.6, hatch='//', label='Val Pass'),
        Patch(facecolor='grey', alpha=0.3, hatch='xx', label='Data Fetch'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

def plot_time_metrics_finetune(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    do_stacked_time_plot(ax, df, "strategy", ["linear_probe", "last_block", "selective_20pct", "full"], "Time Breakdown: Fine-Tuning")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "finetune_time_breakdown.png"))
    plt.close(fig)

def plot_time_metrics_fewshot(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    do_stacked_time_plot(ax, df, "pct", [5, 20, 100], "Time Breakdown: Few-Shot Regimes")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fewshot_time_breakdown.png"))
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
    ax.set_title("Fine-Tuning Strategy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "finetune_comparison.png"))
    plt.close(fig)

def plot_cross_model_fewshot(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    pcts = [5, 20, 100]
    x = np.arange(len(pcts))
    width = 0.25
    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model]
        vals = [sub[sub["pct"] == p]["best_val_acc"].values[0] if not sub[sub["pct"] == p].empty else 0 for p in pcts]
        ax.bar(x + (i-1) * width, vals, width, label=model, color=PALETTE[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["5%", "20%", "100%"])
    ax.set_xlabel("Training Data %")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Few-Shot Performance")
    ax.legend(loc='lower right')
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fewshot_comparison.png"))
    plt.close(fig)

def plot_cross_model_layer_probe(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model].copy()
        sub["depth_order"] = sub["depth"].map({"early": 0, "mid": 1, "final": 2})
        sub = sub.sort_values("depth_order")
        ax.plot(["Early", "Mid", "Final"], sub["best_val_acc"].values,
                marker="o", linewidth=2, label=model, color=PALETTE[i])
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Layer Probe Accuracy vs Depth")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "layer_probe_comparison.png"))
    plt.close(fig)

def plot_corruption_heatmap(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    corruptions = ["gaussian_0.05", "gaussian_0.10", "gaussian_0.20", "motion_blur", "brightness_dark", "brightness_bright"]
    x = np.arange(len(corruptions))
    width = 0.25
    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model]
        if sub.empty: continue
        vals = []
        for c in corruptions:
            row = sub[sub["corruption"] == c]
            if not row.empty:
                vals.append(row["corrupted_acc"].values[0])
            else:
                vals.append(0)
        ax.bar(x + (i-1) * width, vals, width, label=model, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Gauss 0.05", "Gauss 0.1", "Gauss 0.2", "Blur", "Dark", "Bright"], rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness Under Corruption")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "corruption_bar_chart.png"))
    plt.close(fig)

def plot_training_dynamics(results_root: str, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, model in enumerate(MODELS):
        history = load_history(results_root, model, "scenario1_linear_probe")
        if not history: continue
        ax = axes[i]
        ax.plot(history["train_loss"], label="Train Loss", color='blue')
        ax.set_ylabel("Loss", color='blue')
        ax2 = ax.twinx()
        ax2.plot(history["val_acc"], label="Val Acc", color='red')
        ax2.set_ylabel("Val Accuracy (%)", color='red')
        ax.set_xlabel("Epochs")
        ax.set_title(f"{model} (Linear Probe)")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_dynamics_comparison.png"))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    rr = args.results_root
    agg_dir = os.path.join(rr, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    print("\n=== Aggregating Comprehensive Results ===\n")

    dfs = {
        "efficiency": aggregate_efficiency(rr),
        "scenario1": aggregate_scenario1(rr),
        "finetune": aggregate_finetune(rr),
        "fewshot": aggregate_fewshot(rr),
        "corruption": aggregate_corruption(rr),
        "layer_probe": aggregate_layer_probe(rr),
    }

    for name, df in dfs.items():
        if not df.empty:
            df.to_csv(os.path.join(agg_dir, f"{name}_detailed.csv"), index=False)
            
    if not dfs["finetune"].empty:
        plot_time_metrics_finetune(dfs["finetune"], agg_dir)
        plot_cross_model_finetune(dfs["finetune"], agg_dir)
        
    if not dfs["fewshot"].empty:
        plot_time_metrics_fewshot(dfs["fewshot"], agg_dir)
        plot_cross_model_fewshot(dfs["fewshot"], agg_dir)
        
    if not dfs["layer_probe"].empty:
        plot_cross_model_layer_probe(dfs["layer_probe"], agg_dir)
        
    if not dfs["corruption"].empty:
        plot_corruption_heatmap(dfs["corruption"], agg_dir)

    plot_training_dynamics(rr, agg_dir)
    print("Graphs generated successfully.")

if __name__ == "__main__":
    main()
