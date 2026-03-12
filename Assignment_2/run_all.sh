#!/bin/bash
# GNR 638 Assignment 2 — Run all sub-scenarios
# ================================================
# Each line is an INDEPENDENT run for one model × scenario × sub-scenario.
# You can run them one at a time or in sequence.
# Recommended: run from the Assignment_2/ directory with conda env activated.
#
# conda activate gnr638_ass
# bash run_all.sh
#
# To run only one model, comment out the others.

set -e  # Stop on error
PYTHON="python"

echo "========================================================"
echo " GNR 638 Assignment 2 — Automated Run Script"
echo "========================================================"

# ── SCENARIO 1: Linear Probe ──────────────────────────────────────────────────
echo -e "\n[S1] Linear Probe"
for MODEL in resnet50 inception_v3 densenet121; do
    echo "  → $MODEL"
    $PYTHON train.py --model $MODEL --scenario 1
done

# ── SCENARIO 2: Fine-Tuning Strategies ────────────────────────────────────────
echo -e "\n[S2] Fine-Tuning Strategies"
for MODEL in resnet50 inception_v3 densenet121; do
    for STRATEGY in linear_probe last_block selective_20pct full; do
        echo "  → $MODEL / $STRATEGY"
        $PYTHON train.py --model $MODEL --scenario 2 --strategy $STRATEGY
    done
done

# ── SCENARIO 3: Few-Shot Learning ─────────────────────────────────────────────
echo -e "\n[S3] Few-Shot Learning"
for MODEL in resnet50 inception_v3 densenet121; do
    for PCT in 100 20 5; do
        echo "  → $MODEL / $PCT%"
        $PYTHON train.py --model $MODEL --scenario 3 --pct $PCT
    done
done

# ── SCENARIO 4: Corruption Robustness ─────────────────────────────────────────
echo -e "\n[S4] Corruption Robustness"
for MODEL in resnet50 inception_v3 densenet121; do
    for CORRUPTION in gaussian_0.05 gaussian_0.10 gaussian_0.20 motion_blur brightness_dark brightness_bright; do
        echo "  → $MODEL / $CORRUPTION"
        $PYTHON train.py --model $MODEL --scenario 4 --corruption $CORRUPTION
    done
done

# ── SCENARIO 5: Layer-Wise Feature Probing ────────────────────────────────────
echo -e "\n[S5] Layer-Wise Feature Probing"
for MODEL in resnet50 inception_v3 densenet121; do
    for DEPTH in early mid final; do
        echo "  → $MODEL / $DEPTH"
        $PYTHON train.py --model $MODEL --scenario 5 --depth $DEPTH
    done
done

# ── Aggregate all results ─────────────────────────────────────────────────────
echo -e "\n[AGG] Aggregating results..."
$PYTHON analyze_results.py

echo -e "\n✓ All scenarios completed! Results in results/"
