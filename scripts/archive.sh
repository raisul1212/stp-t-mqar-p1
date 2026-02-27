#!/bin/bash
# ================================================================
# archive.sh — Package all results for download before stopping pod
# ================================================================
# Creates a single tarball with all results, checkpoints, and logs.
# Safe to run multiple times (uses timestamps).
#
# Usage:
#   bash stp-t-mqar-p1/scripts/archive.sh
# ================================================================

WORKSPACE="${WORKSPACE:-/workspace}"
TIMESTAMP=$(date +%Y%m%d_%H%M)
ARCHIVE="$WORKSPACE/stp_mqar_p1_${TIMESTAMP}.tar.gz"

echo "============================================"
echo "Archiving STP-T MQAR Phase 1 Results"
echo "============================================"

# Run extraction first (if not already done)
if [ -f "$WORKSPACE/extract_results.py" ]; then
    echo "Running result extraction..."
    python3 "$WORKSPACE/extract_results.py" --output_dir "$WORKSPACE/results" 2>&1
    echo ""
fi

# Count what we have
N_RUNS=$(ls "$WORKSPACE/results/runs/"*.json 2>/dev/null | wc -l)
N_CKPTS=$(find "$WORKSPACE/checkpoints" -name "*.pt" 2>/dev/null | wc -l)
N_WANDB=$(ls -d "$WORKSPACE/zoology/wandb/offline-run-"* 2>/dev/null | wc -l)

echo "Contents:"
echo "  Per-run JSONs:   $N_RUNS"
echo "  Checkpoints:     $N_CKPTS"
echo "  WandB runs:      $N_WANDB"
echo ""

# Build tarball
echo "Creating archive..."
tar czf "$ARCHIVE" \
    -C "$WORKSPACE" \
    results/ \
    checkpoints/ \
    experiment_log.txt \
    zoology/wandb/ \
    stp-t-mqar-p1/ \
    2>/dev/null

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo ""
echo "============================================"
echo "Archive: $ARCHIVE ($SIZE)"
echo "============================================"
echo ""
echo "Download with:"
echo "  runpodctl receive $ARCHIVE"
echo "  # or scp from your local machine"
echo ""
echo "Safe to stop the pod now."
