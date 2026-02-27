#!/bin/bash
# ================================================================
# run.sh — Run MQAR benchmark with automated result saving
# ================================================================
# Model selection uses the SAME STP_MODELS env var as mqar_p1.py.
# This script just wraps: launch → extract → archive.
#
# Usage:
#   # Run only STP models (~2-4h)
#   bash run.sh stp_v3,stp_v4
#
#   # STP vs direct competitors (~6-8h)
#   bash run.sh stp_v3,stp_v4,retnet,gla
#
#   # Everything (~24-48h)
#   bash run.sh
#   bash run.sh all
#
#   # Skip checkpoints (faster, less disk)
#   STP_SAVE_CHECKPOINTS=none bash run.sh stp_v3,stp_v4
#
# The first argument is the model filter (same names as STP_MODELS).
# If omitted or "all", runs everything.
# ================================================================

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/stp-t-mqar-p1"

# ── Model selection: first arg or STP_MODELS env var ──
if [ -n "$1" ]; then
    export STP_MODELS="$1"
elif [ -z "$STP_MODELS" ]; then
    export STP_MODELS="all"
fi

# ── Environment ──
export WANDB_MODE=offline
export STP_RESULTS_DIR="${STP_RESULTS_DIR:-$WORKSPACE/results}"
export STP_CHECKPOINT_DIR="${STP_CHECKPOINT_DIR:-$WORKSPACE/checkpoints}"
export STP_SAVE_CHECKPOINTS="${STP_SAVE_CHECKPOINTS:-best}"

echo "============================================"
echo "STP-T MQAR Phase 1"
echo "============================================"
echo "Models:         ${STP_MODELS}"
echo "Results dir:    $STP_RESULTS_DIR"
echo "Checkpoints:    $STP_CHECKPOINT_DIR (mode: $STP_SAVE_CHECKPOINTS)"
echo "============================================"
echo ""
echo "Available model names:"
echo "  attention, based, mamba2, delta_net, gated_delta_net,"
echo "  gla, retnet, stp_v3, stp_v4"
echo ""

cd "$WORKSPACE/zoology"

# ── Run sweep (STP_MODELS is read inside mqar_p1.py) ──
echo "[1/3] Running experiment sweep (models: ${STP_MODELS})..."
python3 -m zoology.launch "$REPO_DIR/configs/mqar_p1.py" 2>&1 | tee "$WORKSPACE/experiment_log.txt"

# ── Extract results ──
echo ""
echo "[2/3] Extracting and aggregating results..."
python3 "$WORKSPACE/extract_results.py" --output_dir "$STP_RESULTS_DIR" 2>&1

# ── Archive ──
echo ""
echo "[3/3] Archiving..."
bash "$REPO_DIR/scripts/archive.sh"

echo ""
echo "============================================"
echo "ALL DONE (models: ${STP_MODELS})"
echo "============================================"
echo ""
echo "Key files:"
echo "  $STP_RESULTS_DIR/comparison_table.txt  ← Quick look at results"
echo "  $STP_RESULTS_DIR/best_per_model.json   ← Best configs (for paper)"
echo "  $STP_RESULTS_DIR/summary.csv           ← All runs (for analysis)"
echo "  $STP_CHECKPOINT_DIR/                   ← Model weights"
echo ""
