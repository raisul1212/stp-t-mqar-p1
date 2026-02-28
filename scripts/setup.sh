#!/bin/bash
# ================================================================
# STP-T MQAR Phase 1: Setup Script (v3)
# ================================================================
#
# DESIGN PRINCIPLE:
#   RunPod pods ship with a matched PyTorch/torchvision/torchaudio
#   stack. We NEVER let pip modify it. All third-party packages are
#   installed with --no-deps, and their sub-packages are installed
#   individually with --no-deps too.
#
# What this script does:
#   1. Verifies GPU + PyTorch (pre-installed, never touched)
#   2. Installs missing Python packages (--no-deps where risky)
#   3. Installs flash-linear-attention + fla-core (--no-deps)
#   4. Clones/installs Zoology (--no-deps)
#   5. Patches Zoology bugs (return_embeddings, pydantic v2)
#   6. Installs STP mixer + patched train.py
#   7. Verifies everything
#
# Usage:
#   bash stp-t-mqar-p1/scripts/setup.sh
# ================================================================

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/stp-t-mqar-p1"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo "============================================"
echo "STP-T MQAR Phase 1 Setup (v3)"
echo "============================================"

# ── 1. GPU ──
echo "[1/7] GPU check..."
if ! nvidia-smi &>/dev/null; then
    fail "No GPU detected"; exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "$GPU_NAME"

# ── 2. System packages ──
echo "[2/7] System packages..."
apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1 && ok "tmux" || warn "tmux failed"

# ── 3. Python deps ──
echo "[3/7] Python dependencies..."

# Record the pod's torch version — we protect this
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "$TORCH_VER" = "none" ]; then
    fail "PyTorch not found. Use a PyTorch pod template."; exit 1
fi
ok "PyTorch $TORCH_VER (pod-installed, will not modify)"

# Safe packages (pure Python, no torch dependency conflicts)
for pkg in wandb pandas tqdm einops pyyaml rich; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null || \
        pip install "$pkg" --break-system-packages -q 2>/dev/null
done

# pydantic — need v2 but must not pull in new torch
python3 -c "import pydantic" 2>/dev/null || \
    pip install pydantic --break-system-packages -q --no-deps 2>/dev/null

# transformers — already on most pods, install --no-deps if missing
python3 -c "import transformers" 2>/dev/null || \
    pip install transformers --break-system-packages -q --no-deps 2>/dev/null

# opt-einsum / scipy
python3 -c "import opt_einsum" 2>/dev/null || \
    pip install opt-einsum --break-system-packages -q 2>/dev/null
python3 -c "import scipy" 2>/dev/null || \
    pip install scipy --break-system-packages -q 2>/dev/null

ok "Python packages"

# ── flash-linear-attention ──
# Install BOTH fla and fla-core with --no-deps to avoid pulling new torch.
# fla 0.4.1+ has exist_ok=True in AutoConfig.register (fixes BitNet conflict).
FLA_OK=0
if python3 -c "from fla.layers import GatedLinearAttention" 2>/dev/null; then
    ok "flash-linear-attention (existing, import OK)"
    FLA_OK=1
fi

if [ "$FLA_OK" -eq 0 ]; then
    # Install both packages with --no-deps so torch is untouched
    pip install "flash-linear-attention>=0.4.1" --break-system-packages -q --no-deps --force-reinstall 2>/dev/null
    pip install fla-core --break-system-packages -q --no-deps 2>/dev/null
    ok "flash-linear-attention + fla-core installed (--no-deps)"

    if python3 -c "from fla.layers import GatedLinearAttention" 2>/dev/null; then
        ok "fla import verified"
        FLA_OK=1
    else
        # Fallback: patch BitNet __init__.py to add exist_ok=True
        warn "fla import failing — attempting BitNet registration patch"
        FLA_BITNET=$(python3 -c "import fla, os; print(os.path.join(os.path.dirname(fla.__file__), 'models', 'bitnet', '__init__.py'))" 2>/dev/null)
        if [ -n "$FLA_BITNET" ] && [ -f "$FLA_BITNET" ]; then
            sed -i 's/\.register(\([^)]*\))/\.register(\1, exist_ok=True)/g' "$FLA_BITNET"
            sed -i 's/exist_ok=True, exist_ok=True/exist_ok=True/g' "$FLA_BITNET"
            if python3 -c "from fla.layers import GatedLinearAttention" 2>/dev/null; then
                ok "fla import fixed (patched BitNet registration)"
                FLA_OK=1
            else
                warn "fla still broken — FLA baselines will be skipped"
            fi
        else
            warn "Could not locate fla — FLA baselines will be skipped"
        fi
    fi
fi

# Verify torch wasn't changed
TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ "$TORCH_VER" != "$TORCH_AFTER" ]; then
    fail "PyTorch version changed ($TORCH_VER → $TORCH_AFTER)! Reinstall pod torch."
    exit 1
fi
ok "PyTorch version preserved: $TORCH_VER"

# ── 4. Zoology ──
echo "[4/7] Zoology framework..."
if [ ! -d "$WORKSPACE/zoology" ]; then
    cd "$WORKSPACE"
    git clone -q https://github.com/HazyResearch/zoology.git
    ok "Cloned Zoology"
fi

cd "$WORKSPACE/zoology"
if ! python3 -c "import zoology" 2>/dev/null; then
    pip install -e . --break-system-packages -q --no-deps 2>/dev/null \
        && ok "Zoology installed (--no-deps)" || { fail "Zoology install failed"; exit 1; }
else
    ok "Zoology (existing)"
fi

# ── Zoology patches ──

# Fix 1: return_embeddings=True → False (without this, all models plateau at 25%)
if grep -q "return_embeddings=True" zoology/model.py 2>/dev/null; then
    sed -i 's/return_embeddings=True/return_embeddings=False/' zoology/model.py
    ok "Patched: return_embeddings=False"
else
    ok "Patch already applied: return_embeddings=False"
fi

# Fix 2: pydantic v2 compatibility for LoggerConfig.
# Zoology defines `project_name: str = None` which pydantic v2 rejects.
# Previous manual fixes may have set it to `str = ""` which also breaks
# (WandbLogger checks `is None` to skip logging).
# Correct target: `Optional[str] = None`
#
# Strategy: test if it works. If not, rewrite the lines regardless of current state.
PYDANTIC_OK=$(python3 -c "
from zoology.config import LoggerConfig
lc = LoggerConfig()
assert lc.project_name is None and lc.entity is None
print('ok')
" 2>/dev/null || echo "fail")

if [ "$PYDANTIC_OK" = "ok" ]; then
    ok "Patch already applied: pydantic v2 LoggerConfig"
else
    # Ensure Optional is in the typing import line (not just anywhere in the file)
    if ! grep "from typing import" zoology/config.py | grep -q "Optional"; then
        sed -i 's/from typing import \(.*\)/from typing import Optional, \1/' zoology/config.py
    fi
    # Replace whatever the current project_name/entity lines are
    sed -i 's/project_name:.*$/project_name: Optional[str] = None/' zoology/config.py
    sed -i 's/entity:.*$/entity: Optional[str] = None/' zoology/config.py
    # Verify
    PYDANTIC_CHECK=$(python3 -c "
from zoology.config import LoggerConfig
lc = LoggerConfig()
assert lc.project_name is None and lc.entity is None
print('ok')
" 2>/dev/null || echo "fail")
    if [ "$PYDANTIC_CHECK" = "ok" ]; then
        ok "Patched: pydantic v2 LoggerConfig"
    else
        fail "Could not fix LoggerConfig — check zoology/config.py manually"
        exit 1
    fi
fi

# ── 5. STP mixer ──
echo "[5/7] STP-T mixer..."
if [ ! -d "$REPO_DIR" ]; then
    fail "stp-t-mqar-p1 not found at $REPO_DIR"
    echo "  Clone it first: cd $WORKSPACE && git clone <repo-url> stp-t-mqar-p1"
    exit 1
fi
cp "$REPO_DIR/mixers/stp_zoology.py" "$WORKSPACE/zoology/zoology/mixers/stp.py"
ok "STP mixer installed"

# ── 6. Patch train.py for results automation ──
echo "[6/7] Patching train.py for results automation..."
if [ -f "$REPO_DIR/scripts/stp_train.py" ]; then
    if [ ! -f "$WORKSPACE/zoology/zoology/train.py.orig" ]; then
        cp "$WORKSPACE/zoology/zoology/train.py" "$WORKSPACE/zoology/zoology/train.py.orig"
    fi
    cp "$REPO_DIR/scripts/stp_train.py" "$WORKSPACE/zoology/zoology/train.py"
    ok "train.py patched (checkpoints + JSON results + best-epoch tracking)"
else
    warn "stp_train.py not found — using original train.py (no checkpoints)"
fi

if [ -f "$REPO_DIR/scripts/extract_results.py" ]; then
    cp "$REPO_DIR/scripts/extract_results.py" "$WORKSPACE/extract_results.py"
    ok "extract_results.py installed"
fi

mkdir -p "$WORKSPACE/results/runs"
mkdir -p "$WORKSPACE/checkpoints"
ok "Results dirs: $WORKSPACE/results/, $WORKSPACE/checkpoints/"

# ── 7. Verify ──
echo "[7/7] Verification..."
cd "$WORKSPACE/zoology"
python3 << 'PYEOF'
import torch, sys, os

# GPU
if not torch.cuda.is_available():
    print("  ✗ CUDA not available!"); sys.exit(1)
print(f"  ✓ PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")

# STP
from zoology.mixers.stp import STPAttention, STPAttentionV4
print("  ✓ STP-T v3/v4 importable")

# FLA
try:
    from fla.layers import GatedLinearAttention, DeltaNet, MultiScaleRetention
    print("  ✓ FLA baselines (GLA, DeltaNet, RetNet)")
except Exception as e:
    print(f"  ⚠ FLA not available: {e}")

# Zoology MHA
try:
    from zoology.mixers.attention import MHA
    print("  ✓ Zoology MHA")
except Exception as e:
    print(f"  ⚠ Zoology MHA: {e}")

# GPU forward pass (STP only — fastest check)
B, T, D = 2, 32, 64
x = torch.randn(B, T, D).cuda()
for name, cls in [("STPv3", STPAttention), ("STPv4", STPAttentionV4)]:
    m = cls(d_model=D, num_heads=2).cuda()
    y = m(x)
    assert y.shape == (B, T, D), f"{name} output shape mismatch"
print("  ✓ GPU forward passes OK")

# Logits check (verifies return_embeddings fix)
from zoology.config import ModelConfig, ModuleConfig
from zoology.model import LanguageModel
mixer = ModuleConfig(name="zoology.mixers.attention.MHA", kwargs={})
model = LanguageModel(ModelConfig(
    vocab_size=8192, d_model=64, n_layers=2,
    max_position_embeddings=0, sequence_mixer=mixer,
    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
)).cuda()
out = model(torch.randint(0, 8192, (2, 64)).cuda())
assert out.shape == (2, 64, 8192), f"Logits shape {out.shape} != (2, 64, 8192)"
print("  ✓ Logits shape correct (return_embeddings fix working)")

# Config loading check (verifies pydantic v2 fix)
from zoology.config import LoggerConfig
try:
    lc = LoggerConfig()
    assert lc.project_name is None
    print("  ✓ LoggerConfig() accepts None (pydantic v2 fix working)")
except Exception as e:
    print(f"  ✗ LoggerConfig failed: {e}"); sys.exit(1)

# train.py patch check
from zoology.train import Trainer
if hasattr(Trainer, '_save_checkpoint') and hasattr(Trainer, '_save_run_results'):
    print("  ✓ train.py patched (checkpoint + JSON results)")
else:
    print("  ⚠ train.py NOT patched")

# Results dirs
for d in ["/workspace/results/runs", "/workspace/checkpoints"]:
    if os.path.isdir(d):
        print(f"  ✓ {d}")
    else:
        print(f"  ⚠ {d} missing")

print("\n  ALL CHECKS PASSED")
PYEOF

echo ""
echo "============================================"
echo "SETUP COMPLETE"
echo "============================================"
echo ""
echo "Quick start:"
echo "  tmux new -s stp"
echo "  bash $REPO_DIR/scripts/run.sh stp_v3,stp_v4          # STP only (~2-4h)"
echo "  bash $REPO_DIR/scripts/run.sh stp_v3,stp_v4,retnet,gla  # vs competitors (~6-8h)"
echo "  bash $REPO_DIR/scripts/run.sh                          # everything (~24-48h)"
echo ""
