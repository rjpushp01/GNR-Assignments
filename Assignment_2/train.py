"""
GNR 638 Assignment 2 — Unified Training Entry Point
=====================================================

Each run is a focused, short execution for ONE model + ONE scenario + ONE sub-scenario.
Results are stored in dedicated sub-directories.

Examples:
    # Scenario 1: Linear Probe
    python train.py --model resnet50 --scenario 1
    python train.py --model inception_v3 --scenario 1
    python train.py --model densenet121 --scenario 1

    # Scenario 2: Fine-Tuning Strategies (run separately per strategy)
    python train.py --model resnet50 --scenario 2 --strategy linear_probe
    python train.py --model resnet50 --scenario 2 --strategy last_block
    python train.py --model resnet50 --scenario 2 --strategy full
    python train.py --model resnet50 --scenario 2 --strategy selective_20pct

    # Scenario 3: Few-Shot Learning (run separately per data %)
    python train.py --model resnet50 --scenario 3 --pct 100
    python train.py --model resnet50 --scenario 3 --pct 20
    python train.py --model resnet50 --scenario 3 --pct 5

    # Scenario 4: Corruption Robustness (run separately per corruption)
    python train.py --model resnet50 --scenario 4 --corruption gaussian_0.05
    python train.py --model resnet50 --scenario 4 --corruption gaussian_0.10
    python train.py --model resnet50 --scenario 4 --corruption gaussian_0.20
    python train.py --model resnet50 --scenario 4 --corruption motion_blur
    python train.py --model resnet50 --scenario 4 --corruption brightness_dark
    python train.py --model resnet50 --scenario 4 --corruption brightness_bright

    # Scenario 5: Layer-Wise Feature Probing (run separately per depth)
    python train.py --model resnet50 --scenario 5 --depth early
    python train.py --model resnet50 --scenario 5 --depth mid
    python train.py --model resnet50 --scenario 5 --depth final
"""

import argparse
import os
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNR 638 Assignment 2 — CNN Transfer Learning on AID Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["resnet50", "inception_v3", "densenet121"],
        help="CNN backbone to use",
    )
    parser.add_argument(
        "--scenario", type=int, required=True,
        choices=[1, 2, 3, 4, 5],
        help="Experimental scenario number",
    )
    parser.add_argument(
        "--data_root", type=str, default="train_data",
        help="Path to AID dataset root (default: train_data/)",
    )
    parser.add_argument(
        "--results_root", type=str, default="results",
        help="Root directory for results (default: results/)",
    )

    # Scenario 2 arguments
    parser.add_argument(
        "--strategy", type=str,
        choices=["linear_probe", "last_block", "full", "selective_20pct"],
        default="full",
        help="[Scenario 2] Fine-tuning strategy",
    )

    # Scenario 3 arguments
    parser.add_argument(
        "--pct", type=int,
        choices=[5, 20, 100],
        default=100,
        help="[Scenario 3] Percentage of training data to use",
    )

    # Scenario 4 arguments
    parser.add_argument(
        "--corruption", type=str,
        choices=["gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
                 "motion_blur", "brightness_dark", "brightness_bright"],
        default="gaussian_0.10",
        help="[Scenario 4] Corruption type",
    )

    # Scenario 5 arguments
    parser.add_argument(
        "--depth", type=str,
        choices=["early", "mid", "final"],
        default="final",
        help="[Scenario 5] Layer depth for feature probing",
    )

    return parser.parse_args()


def get_save_dir(args) -> str:
    """Determine the results output directory based on scenario and sub-scenario."""
    base = os.path.join(args.results_root, args.model)
    if args.scenario == 1:
        return os.path.join(base, "scenario1_linear_probe")
    elif args.scenario == 2:
        return os.path.join(base, "scenario2_finetuning", args.strategy)
    elif args.scenario == 3:
        return os.path.join(base, "scenario3_few_shot", f"pct_{args.pct}")
    elif args.scenario == 4:
        return os.path.join(base, "scenario4_corruption", args.corruption)
    elif args.scenario == 5:
        return os.path.join(base, "scenario5_layer_probe", args.depth)
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")


def main():
    args = parse_args()

    # ── Device setup ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  ⚠ CUDA not available — running on CPU")
    print(f"  Device: {device}")

    # ── Resolve paths ─────────────────────────────────────────────────────
    data_root = os.path.abspath(args.data_root)
    save_dir = os.path.abspath(get_save_dir(args))
    print(f"  Data root:  {data_root}")
    print(f"  Save dir:   {save_dir}")

    if not os.path.isdir(data_root):
        print(f"\n  ✗ ERROR: data_root '{data_root}' not found!")
        sys.exit(1)

    # ── Dispatch to scenario ──────────────────────────────────────────────
    if args.scenario == 1:
        from scenarios.scenario1_linear_probe import run
        run(args.model, data_root, save_dir, device)

    elif args.scenario == 2:
        from scenarios.scenario2_finetune import run
        run(args.model, data_root, save_dir, device, strategy=args.strategy)

    elif args.scenario == 3:
        from scenarios.scenario3_fewshot import run
        run(args.model, data_root, save_dir, device, pct=args.pct)

    elif args.scenario == 4:
        from scenarios.scenario4_corruption import run
        run(args.model, data_root, save_dir, device, corruption=args.corruption)

    elif args.scenario == 5:
        from scenarios.scenario5_layer_probe import run
        run(args.model, data_root, save_dir, device, depth=args.depth)

    print(f"\n  ✓ Completed: {args.model} — Scenario {args.scenario}")


if __name__ == "__main__":
    main()
