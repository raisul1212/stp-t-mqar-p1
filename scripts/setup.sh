#!/bin/bash
# ================================================================
# STP-T MQAR Phase 1: Setup Script (v2 — with results automation)
# ================================================================
# Installs Zoology + deps, copies STP mixer, patches train.py for:
#   1. Model checkpoint saving (best + final)
#   2. Best-epoch tracking
#   3. Per-run JSON result export
#   4. Automated post-sweep extraction
#
# Usage:
#   bash setup.sh
# ================================================================

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/stp-t-mqar-p1"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo "============================================"
echo "STP-T MQAR Phase 1 Setup (v2)"
echo "============================================"

# ── GPU ──
echo "[1/7] GPU check..."
if ! nvidia-smi &>/dev/null; then
    fail "No GPU detected"; exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "$GPU_NAME"

# ── System packages ──
echo "[2/7] System packages..."
apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1 && ok "tmux" || warn "tmux failed"

# ── Python deps ──
echo "[3/7] Python dependencies..."
PIP="pip install --break-system-packages -q"

ROOT_FREE=$(df / 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')
if [ -n "$ROOT_FREE" ] && [ "$ROOT_FREE" -lt 8 ]; then
    PIP_TARGET="$WORKSPACE/pylibs"
    mkdir -p "$PIP_TARGET"
    PIP="$PIP --target=$PIP_TARGET"
    export PYTHONPATH="$PIP_TARGET:$PYTHONPATH"
    warn "Root disk small (${ROOT_FREE}GB), installing to $PIP_TARGET"
fi

for pkg in transformers pydantic wandb pandas tqdm einops opt-einsum scipy pyyaml; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null || $PIP "$pkg" 2>/dev/null
done
ok "Python packages"

# Torch
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    ok "PyTorch $TORCH_VER (existing)"
else
    echo "  Installing PyTorch..."
    pip install torch torchvision torchaudio --break-system-packages -q \
        && ok "PyTorch installed" || { fail "PyTorch failed"; exit 1; }
fi

# fla (for GLA, DeltaNet, Gated DeltaNet, Mamba2, RetNet)
# MUST get 0.4.1+ which has exist_ok=True in AutoConfig.register.
# Older versions crash on pods where transformers already registers BitNet.
FLA_OK=0
if python3 -c "from fla.layers import GatedLinearAttention" 2>/dev/null; then
    ok "flash-linear-attention (existing, import OK)"
    FLA_OK=1
fi

if [ "$FLA_OK" -eq 0 ]; then
    pip install "flash-linear-attention>=0.4.1" fla-core --break-system-packages -q --force-reinstall \
        && ok "flash-linear-attention installed (forced 0.4.1+)" \
        || warn "fla pip install failed"

    # Verify import works
    if python3 -c "from fla.layers import GatedLinearAttention" 2>/dev/null; then
        ok "fla import verified"
        FLA_OK=1
    else
        # Fallback: patch BitNet __init__.py to add exist_ok=True
        warn "fla import failing — attempting BitNet registration patch"
        FLA_BITNET=$(python3 -c "import fla; import os; print(os.path.join(os.path.dirname(fla.__file__), 'models', 'bitnet', '__init__.py'))" 2>/dev/null)
        if [ -n "$FLA_BITNET" ] && [ -f "$FLA_BITNET" ]; then
            # Add exist_ok=True to all .register() calls that lack it
            sed -i 's/\.register(\([^)]*\))/\.register(\1, exist_ok=True)/g' "$FLA_BITNET"
            # Clean up any doubled exist_ok from re-running
            sed -i 's/exist_ok=True, exist_ok=True/exist_ok=True/g' "$FLA_BITNET"

            if python3 -c "from fla.layers import GatedLinearAttention" 2>/dev/null; then
                ok "fla import fixed (patched BitNet registration)"
                FLA_OK=1
            else
                warn "fla still broken — FLA baselines (RetNet, GLA, DeltaNet, Mamba2) will be skipped"
            fi
        else
            warn "Could not locate fla BitNet __init__.py — FLA baselines will be skipped"
        fi
    fi
fi

# ── Zoology ──
echo "[4/7] Zoology framework..."
if [ ! -d "$WORKSPACE/zoology" ]; then
    cd "$WORKSPACE"
    git clone -q https://github.com/HazyResearch/zoology.git
    ok "Cloned Zoology"
fi

cd "$WORKSPACE/zoology"
if ! python3 -c "import zoology" 2>/dev/null; then
    pip install -e . --break-system-packages -q --no-deps 2>/dev/null \
        && ok "Zoology installed" || { fail "Zoology install failed"; exit 1; }
else
    ok "Zoology (existing)"
fi

# return_embeddings fix
if grep -q "return_embeddings=True" zoology/model.py 2>/dev/null; then
    sed -i 's/return_embeddings=True/return_embeddings=False/' zoology/model.py
    ok "Applied return_embeddings fix"
else
    ok "return_embeddings fix already applied"
fi

# ── STP mixer ──
echo "[5/7] STP-T mixer..."
if [ ! -d "$REPO_DIR" ]; then
    fail "stp-t-mqar-p1 not found at $REPO_DIR"
    echo "  Clone it first: cd $WORKSPACE && git clone <repo-url> stp-t-mqar-p1"
    exit 1
fi
cp "$REPO_DIR/mixers/stp_zoology.py" "$WORKSPACE/zoology/zoology/mixers/stp.py"
ok "STP mixer installed"

# ── Patch train.py for results automation ──
echo "[6/7] Patching train.py for results automation..."
if [ -f "$REPO_DIR/scripts/stp_train.py" ]; then
    # Back up original
    if [ ! -f "$WORKSPACE/zoology/zoology/train.py.orig" ]; then
        cp "$WORKSPACE/zoology/zoology/train.py" "$WORKSPACE/zoology/zoology/train.py.orig"
    fi
    cp "$REPO_DIR/scripts/stp_train.py" "$WORKSPACE/zoology/zoology/train.py"
    ok "train.py patched (checkpoints + JSON results + best-epoch tracking)"
else
    warn "stp_train.py not found — using original train.py (no checkpoints)"
fi

# Copy extraction script
if [ -f "$REPO_DIR/scripts/extract_results.py" ]; then
    cp "$REPO_DIR/scripts/extract_results.py" "$WORKSPACE/extract_results.py"
    ok "extract_results.py installed"
fi

# Create results directories
mkdir -p "$WORKSPACE/results/runs"
mkdir -p "$WORKSPACE/checkpoints"
ok "Results dirs created: $WORKSPACE/results/, $WORKSPACE/checkpoints/"

# ── Verify ──
echo "[7/7] Verification..."
cd "$WORKSPACE/zoology"
python3 << 'PYEOF'
import torch, sys, os

if not torch.cuda.is_available():
    print("  ✗ CUDA not available!"); sys.exit(1)
print(f"  ✓ PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")

from zoology.mixers.stp import STPAttention, STPAttentionV4
print("  ✓ STP-T v3/v4 importable")

try:
    from fla.layers import GatedLinearAttention, DeltaNet, MultiScaleRetention
    print("  ✓ FLA baselines (GLA, DeltaNet, RetNet)")
except: print("  ⚠ FLA not available")

try:
    from zoology.mixers.attention import MHA
    print("  ✓ Zoology MHA")
except: print("  ⚠ Zoology MHA not found")

# GPU forward pass
B, T, D = 2, 32, 64
x = torch.randn(B, T, D).cuda()
for name, cls in [("STPv3", STPAttention), ("STPv4", STPAttentionV4)]:
    m = cls(d_model=D, num_heads=2).cuda()
    y = m(x)
    assert y.shape == (B, T, D)
print("  ✓ GPU forward passes OK")

# Logits check
from zoology.config import ModelConfig, ModuleConfig
from zoology.model import LanguageModel
mixer = ModuleConfig(name="zoology.mixers.attention.MHA", kwargs={})
model = LanguageModel(ModelConfig(
    vocab_size=8192, d_model=64, n_layers=2,
    max_position_embeddings=0, sequence_mixer=mixer,
    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
)).cuda()
out = model(torch.randint(0, 8192, (2, 64)).cuda())
assert out.shape == (2, 64, 8192)
print("  ✓ Logits shape correct (return_embeddings fix working)")

# Verify train.py patch
from zoology.train import Trainer
if hasattr(Trainer, '_save_checkpoint') and hasattr(Trainer, '_save_run_results'):
    print("  ✓ train.py patched (checkpoint + JSON results)")
else:
    print("  ⚠ train.py NOT patched (no checkpoints, no JSON export)")

# Verify results dirs
for d in ["/workspace/results/runs", "/workspace/checkpoints"]:
    if os.path.isdir(d):
        print(f"  ✓ {d} exists")
    else:
        print(f"  ⚠ {d} missing")

print("\n  ALL CHECKS PASSED")
PYEOF

echo ""
echo "============================================"
echo "SETUP COMPLETE"
echo "============================================"
echo ""
echo "Run ALL benchmarks:"
echo "  tmux new -s stp"
echo "  cd $WORKSPACE/zoology"
echo "  export WANDB_MODE=offline"
echo "  export STP_RESULTS_DIR=$WORKSPACE/results"
echo "  export STP_CHECKPOINT_DIR=$WORKSPACE/checkpoints"
echo "  export STP_SAVE_CHECKPOINTS=best"
echo "  python3 -m zoology.launch $REPO_DIR/configs/mqar_p1.py 2>&1 | tee $WORKSPACE/experiment_log.txt"
echo ""
echo "  # After sweep finishes, extract results:"
echo "  python3 $WORKSPACE/extract_results.py"
echo ""
echo "  # Archive everything (portable):"
echo "  bash $REPO_DIR/scripts/archive.sh"
echo ""
echo "Run SPECIFIC models (use STP_MODELS env var):"
echo "  STP_MODELS=stp_v3,stp_v4       python3 -m zoology.launch $REPO_DIR/configs/mqar_p1.py"
echo "  STP_MODELS=attention            python3 -m zoology.launch $REPO_DIR/configs/mqar_p1.py"
echo "  STP_MODELS=stp_v3,retnet,gla   python3 -m zoology.launch $REPO_DIR/configs/mqar_p1.py"
echo ""
echo "Available model names: attention, based, mamba2, delta_net, gated_delta_net, gla, retnet, stp_v3, stp_v4"
echo ""
echo "RESULTS INFRASTRUCTURE:"
echo "  Per-run JSONs  → $WORKSPACE/results/runs/*.json  (written after each run)"
echo "  Checkpoints    → $WORKSPACE/checkpoints/{run_id}/best.pt, final.pt"
echo "  Aggregated     → $WORKSPACE/results/summary.json, summary.csv, comparison_table.txt"
echo "  WandB offline  → $WORKSPACE/zoology/wandb/"
echo ""
