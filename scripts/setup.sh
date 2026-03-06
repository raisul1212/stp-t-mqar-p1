#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════════
# setup.sh — STP-T MQAR Phase 1 environment setup
# RISE Lab, Purdue University  |  March 2026
#
# PLATFORM SUPPORT:
#   - Cloud GPU pods  (RunPod, Thunder Compute, Lambda Labs, Vast.ai)
#   - Purdue clusters (Anvil, Gilbreth) — set PURDUE=1 or auto-detected
#   - Any Linux machine with Python 3 + PyTorch + CUDA
#
# DESIGN PRINCIPLE:
#   We NEVER let pip modify the pre-installed PyTorch stack.
#   Packages that bundle torch in their deps are installed with
#   --no-deps PLUS their non-torch dependencies are listed explicitly.
#
# KNOWN PLATFORM ISSUES:
#   - Thunder Compute and some Ubuntu 22.04/24.04 images ship Python 3.12
#     with pip 22.0.2, which crashes on ANY operation due to the removed
#     pkgutil.ImpImporter (removed in Python 3.12, but pip 22 still uses it).
#     Both `pip3` and `python3 -m pip` are broken in this state because they
#     load the same /usr/lib/python3/dist-packages/pip code.
#     Fix: bootstrap a fresh pip via get-pip.py before doing anything.
#   - The --break-system-packages flag (pip ≥23.0.1) is not recognized by
#     older pip versions and causes silent failures when errors are swallowed.
#     Fix: detect whether the flag is supported, then use it consistently.
#
# Steps:
#   0.  pip bootstrap (if system pip is broken)
#   1.  Platform detection
#   2.  GPU + PyTorch version check
#   3.  System packages (git, tmux) — skipped gracefully on clusters
#   4.  Safe Python deps (einops, wandb, pyyaml, etc.)
#   5.  flash-linear-attention + fla-core + their non-torch deps
#   6.  Zoology clone + install (--no-deps)
#   7.  Zoology patches (return_embeddings, pydantic v2, fla head_first)
#   8.  Register mixer directory on sys.path (.pth file)
#   9.  Install stp_train.py (checkpoints + JSON results)
#   10. Full verification
# ════════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Helpers (defined FIRST so all steps can use them) ────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

# ── Workspace + repo paths ───────────────────────────────────────────────────
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "${_REPO_ROOT}")}"
REPO_DIR="${_REPO_ROOT}"
ZOOLOGY_DIR="${WORKSPACE}/zoology"
MIXER_DIR="${REPO_DIR}/mixers"

# ── Platform detection ───────────────────────────────────────────────────────
IS_PURDUE=0
if [ "${PURDUE:-0}" = "1" ] || \
   echo "${HOSTNAME:-}" | grep -qiE "anvil|gilbreth|bell|negishi|purdue" || \
   [ -n "${RCAC_CLUSTER:-}" ]; then
    IS_PURDUE=1
fi

# ── Step 0: pip bootstrap + resolution ───────────────────────────────────────
# PROBLEM: Many cloud GPU images (Thunder Compute, some RunPod/Lambda base
# images) ship Ubuntu with Python 3.12 but pip 22.0.2. This pip version
# crashes on ANY operation — including `python3 -m pip install` — because it
# uses pkgutil.ImpImporter which was removed in Python 3.12.
# Both `pip3` and `python3 -m pip` are broken because they load the same
# /usr/lib/python3/dist-packages/pip code.
#
# SOLUTION: Test pip with an actual install operation. If that fails, bootstrap
# a fresh pip via get-pip.py before proceeding. This is safe and idempotent.
#
# NOTE: `python3 -m pip --version` can PASS even when the pip is broken
# (the crash happens deeper in the install codepath), so we must test with
# a real operation like --dry-run.

_pip_works() {
    # Test with an actual operation, not just --version (which can pass on broken pip).
    # Try with --break-system-packages first (PEP 668 systems reject bare --dry-run).
    python3 -m pip install --dry-run --break-system-packages pip &>/dev/null 2>&1 || \
    python3 -m pip install --dry-run pip &>/dev/null 2>&1
}

if ! _pip_works; then
    echo "[0/10] Bootstrapping pip (system pip is broken on this Python)..."
    # Method 1: get-pip.py (most reliable, works even with no pip at all)
    if command -v curl &>/dev/null; then
        curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        python3 /tmp/get-pip.py --quiet 2>&1 | tail -3
    elif command -v wget &>/dev/null; then
        wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
        python3 /tmp/get-pip.py --quiet 2>&1 | tail -3
    # Method 2: ensurepip (available on some images)
    elif python3 -m ensurepip --upgrade &>/dev/null; then
        python3 -m pip install --upgrade pip -q 2>/dev/null || true
    else
        fail "Cannot bootstrap pip. Install pip manually and rerun."
        exit 1
    fi

    if _pip_works; then
        ok "pip bootstrapped: $(python3 -m pip --version 2>/dev/null)"
    else
        fail "pip bootstrap failed. Install pip manually and rerun."
        exit 1
    fi
else
    ok "pip OK: $(python3 -m pip --version 2>/dev/null)"
fi

PIP="python3 -m pip"

# Detect whether --break-system-packages is supported (pip ≥23.0.1).
# This flag is needed on externally-managed Python installs (PEP 668) but
# is an unknown option on older pip, causing hard failures.
BSP=""
if ${PIP} install --dry-run --break-system-packages pip &>/dev/null 2>&1; then
    BSP="--break-system-packages"
fi

# On Purdue outside conda: add --user since system site-packages is read-only
PIP_EXTRA=""
if [ "${IS_PURDUE}" = "1" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    PIP_EXTRA="--user"
fi

# ── pip_install helper ───────────────────────────────────────────────────────
# Central install function so we never silently swallow errors again.
# Usage: pip_install [--no-deps] package1 [package2 ...]
pip_install() {
    local no_deps=""
    if [ "${1:-}" = "--no-deps" ]; then
        no_deps="--no-deps"
        shift
    fi
    ${PIP} install ${BSP} ${PIP_EXTRA} ${no_deps} -q "$@" 2>&1 | \
        grep -v "WARNING: Running pip as the 'root'" || true
}

echo "════════════════════════════════════════════════════════════════"
echo " STP-T MQAR Phase 1 Setup"
echo " Workspace:  ${WORKSPACE}"
echo " Repo:       ${REPO_DIR}"
echo " pip:        $(${PIP} --version 2>/dev/null | head -1)"
if [ "${IS_PURDUE}" = "1" ]; then
echo " Platform:   Purdue cluster (${RCAC_CLUSTER:-${HOSTNAME:-unknown}})"
else
echo " Platform:   Cloud/Linux pod"
fi
echo "════════════════════════════════════════════════════════════════"

# ── 1. Platform-specific pre-checks ─────────────────────────────────────────
echo "[1/10] Platform check..."
if [ "${IS_PURDUE}" = "1" ]; then
    if ! command -v python3 &>/dev/null; then
        fail "python3 not found. On Purdue, load modules first:"
        echo "        module load anaconda"
        echo "        module load cuda"
        echo "        conda activate your_env"
        exit 1
    fi
    ok "Purdue cluster detected — skipping apt-get steps"
    ok "python3: $(python3 --version 2>&1)"
else
    ok "Cloud pod — full setup mode"
fi

# ── 2. GPU check ─────────────────────────────────────────────────────────────
echo "[2/10] GPU check..."
if ! nvidia-smi &>/dev/null; then
    fail "No GPU / nvidia-smi not found"
    if [ "${IS_PURDUE}" = "1" ]; then
        echo "       On Purdue: make sure you have a GPU allocation."
        echo "       Example: salloc -A your_account -p gpu -N 1 --gpus-per-node=1"
    fi
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: ${GPU_NAME} (${GPU_MEM})"

TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "${TORCH_VER}" = "none" ]; then
    fail "PyTorch not found."
    if [ "${IS_PURDUE}" = "1" ]; then
        echo "       Load modules: module load anaconda cuda"
        echo "       Then activate a conda env with PyTorch installed."
    else
        echo "       Use a PyTorch pod template."
    fi
    exit 1
fi
ok "PyTorch ${TORCH_VER}"

# ── 3. System packages ───────────────────────────────────────────────────────
echo "[3/10] System packages..."
if [ "${IS_PURDUE}" = "1" ]; then
    command -v git  &>/dev/null && ok "git (system)" || warn "git not found — install manually"
    command -v tmux &>/dev/null && ok "tmux (system)" || warn "tmux not found — use 'screen' or 'nohup' instead"
else
    if command -v apt-get &>/dev/null; then
        apt-get update -qq 2>/dev/null && \
        apt-get install -y -qq git tmux 2>/dev/null && \
        ok "git, tmux installed" || warn "apt-get failed — git/tmux may already be present"
    else
        warn "apt-get not available — skipping system packages"
    fi
fi

# ── 4. Python dependencies ───────────────────────────────────────────────────
echo "[4/10] Python dependencies..."
# These are safe to install normally (they don't pull in torch).
_DEPS_NEEDED=()
for pkg in einops wandb pyyaml pandas tqdm rich opt-einsum scipy; do
    pymod="${pkg//-/_}"
    if ! python3 -c "import ${pymod}" 2>/dev/null; then
        _DEPS_NEEDED+=("${pkg}")
    fi
done

if [ ${#_DEPS_NEEDED[@]} -gt 0 ]; then
    echo "  Installing: ${_DEPS_NEEDED[*]}"
    pip_install "${_DEPS_NEEDED[@]}"
fi

# pydantic: install normally (its deps are lightweight)
if ! python3 -c "import pydantic" 2>/dev/null; then
    pip_install pydantic
fi

# transformers: install with --no-deps to protect torch,
# then install its key non-torch runtime dependencies explicitly.
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "  Installing: transformers (--no-deps) + non-torch deps"
    pip_install --no-deps transformers
    # transformers runtime deps (excluding torch/tf/jax/flax):
    _TX_DEPS=()
    for txpkg in regex tokenizers safetensors huggingface-hub filelock packaging requests; do
        txmod="${txpkg//-/_}"
        if ! python3 -c "import ${txmod}" 2>/dev/null; then
            _TX_DEPS+=("${txpkg}")
        fi
    done
    if [ ${#_TX_DEPS[@]} -gt 0 ]; then
        pip_install "${_TX_DEPS[@]}"
    fi
fi

# Verify critical imports
_DEP_FAILS=""
for check_mod in einops wandb pydantic transformers; do
    python3 -c "import ${check_mod}" 2>/dev/null || _DEP_FAILS="${_DEP_FAILS} ${check_mod}"
done
if [ -n "${_DEP_FAILS}" ]; then
    fail "Failed to install:${_DEP_FAILS}"
    echo "       Try manually: ${PIP} install${_DEP_FAILS}"
    exit 1
fi
ok "Python deps"

# ── 5. flash-linear-attention ────────────────────────────────────────────────
echo "[5/10] flash-linear-attention..."
FLA_OK=0

if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
    ok "fla (existing, import OK)"
    FLA_OK=1
fi

if [ "${FLA_OK}" -eq 0 ]; then
    # Install fla + fla-core with --no-deps to protect torch.
    # Their non-torch deps (einops, transformers) were already installed above.
    echo "  Installing: flash-linear-attention, fla-core (--no-deps)"
    pip_install --no-deps "flash-linear-attention>=0.4.1"
    pip_install --no-deps fla-core

    if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
        ok "fla installed"
        FLA_OK=1
    else
        # Diagnose: show the actual error instead of swallowing it
        warn "fla import failing. Actual error:"
        python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>&1 | \
            head -5 | while read -r line; do echo "       ${line}"; done

        # Try the BitNet registration patch (fla ≥0.4 sometimes conflicts
        # with transformers' AutoModel registry)
        warn "Attempting BitNet registration patch..."
        FLA_BITNET=$(python3 -c \
            "import fla, os; print(os.path.join(os.path.dirname(fla.__file__), 'models', 'bitnet', '__init__.py'))" \
            2>/dev/null || echo "")
        if [ -n "${FLA_BITNET}" ] && [ -f "${FLA_BITNET}" ]; then
            sed -i 's/\.register(\([^)]*\))/\.register(\1, exist_ok=True)/g' "${FLA_BITNET}"
            sed -i 's/exist_ok=True, exist_ok=True/exist_ok=True/g' "${FLA_BITNET}"
            if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
                ok "fla fixed (patched BitNet registration)"
                FLA_OK=1
            fi
        fi

        # If still broken, try installing whatever module is missing
        if [ "${FLA_OK}" -eq 0 ]; then
            MISSING_MOD=$(python3 -c "
from fla.layers import GatedLinearAttention, MultiScaleRetention
" 2>&1 | grep "No module named" | head -1 | sed "s/.*No module named '\\([^']*\\)'.*/\\1/" || echo "")
            if [ -n "${MISSING_MOD}" ]; then
                warn "Missing module: ${MISSING_MOD} — attempting install"
                pip_install "${MISSING_MOD}" 2>/dev/null || true
                if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
                    ok "fla fixed (installed missing dep: ${MISSING_MOD})"
                    FLA_OK=1
                fi
            fi
        fi

        if [ "${FLA_OK}" -eq 0 ]; then
            warn "fla still broken — RetNet/GLA baselines will fail at runtime"
            warn "  Debug: python3 -c 'from fla.layers import GatedLinearAttention'"
        fi
    fi
fi

# Guard: verify torch was not changed
TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ "${TORCH_VER}" != "${TORCH_AFTER}" ]; then
    fail "PyTorch version changed! (${TORCH_VER} → ${TORCH_AFTER})"
    exit 1
fi
ok "PyTorch version preserved: ${TORCH_VER}"

# ── 6. Zoology ───────────────────────────────────────────────────────────────
echo "[6/10] Zoology framework..."
if [ ! -d "${ZOOLOGY_DIR}" ]; then
    echo "  Cloning Zoology..."
    git clone -q https://github.com/HazyResearch/zoology.git "${ZOOLOGY_DIR}"
    ok "Cloned Zoology → ${ZOOLOGY_DIR}"
fi
cd "${ZOOLOGY_DIR}"
if ! python3 -c "import zoology" 2>/dev/null; then
    pip_install --no-deps -e .
    if python3 -c "import zoology" 2>/dev/null; then
        ok "Zoology installed (--no-deps)"
    else
        fail "Zoology install failed"
        exit 1
    fi
else
    ok "Zoology (existing)"
fi

# ── 7. Zoology patches ───────────────────────────────────────────────────────
echo "[7/10] Zoology patches..."

# Patch A: return_embeddings=False
MODEL_PY="${ZOOLOGY_DIR}/zoology/model.py"
if grep -q "return_embeddings.*=.*True" "${MODEL_PY}" 2>/dev/null; then
    sed -i 's/return_embeddings.*=.*True/return_embeddings: bool = False/g' "${MODEL_PY}"
    ok "Patched: return_embeddings=False"
else
    ok "return_embeddings=False already applied"
fi

# Patch B: pydantic v2 LoggerConfig
PYDANTIC_OK=$(python3 -c "
from zoology.config import LoggerConfig
lc = LoggerConfig()
assert lc.project_name is None and lc.entity is None
print('ok')
" 2>/dev/null || echo "fail")

if [ "${PYDANTIC_OK}" = "ok" ]; then
    ok "pydantic v2 LoggerConfig already OK"
else
    CFG_PY="${ZOOLOGY_DIR}/zoology/config.py"

    # Ensure Optional is imported
    if ! grep "from typing import" "${CFG_PY}" | grep -q "Optional"; then
        sed -i 's/from typing import \(.*\)/from typing import Optional, \1/' "${CFG_PY}"
    fi

    # Use Python for a more robust patch — sed regexes are fragile across
    # different Zoology versions where field formatting varies
    python3 << PYEOF
import re

with open("${CFG_PY}") as f:
    content = f.read()

# Match any line defining project_name or entity in LoggerConfig,
# regardless of type annotation, default value, or whitespace.
content = re.sub(
    r'(project_name)\s*[:=][^\n]*',
    r'project_name: Optional[str] = None',
    content
)
content = re.sub(
    r'(entity)\s*[:=][^\n]*',
    r'entity: Optional[str] = None',
    content
)

with open("${CFG_PY}", "w") as f:
    f.write(content)
PYEOF

    PYDANTIC_CHECK=$(python3 -c "
from zoology.config import LoggerConfig
lc = LoggerConfig()
assert lc.project_name is None and lc.entity is None
print('ok')
" 2>/dev/null || echo "fail")
    if [ "${PYDANTIC_CHECK}" = "ok" ]; then
        ok "Patched: pydantic v2 LoggerConfig"
    else
        fail "Could not fix LoggerConfig — check zoology/config.py manually"
        echo "       Debug: python3 -c 'from zoology.config import LoggerConfig; LoggerConfig()'"
        python3 -c "from zoology.config import LoggerConfig; LoggerConfig()" 2>&1 | \
            head -5 | while read -r line; do echo "       ${line}"; done
        exit 1
    fi
fi

# Patch C: fla head_first compatibility
echo "  Patching fla head_first compatibility..."
for fla_wrapper in gla.py delta_net.py gated_delta_net.py rwkv7.py; do
    fpath="${ZOOLOGY_DIR}/zoology/mixers/${fla_wrapper}"
    [ ! -f "${fpath}" ] && continue
    if grep -q "head_first" "${fpath}" 2>/dev/null; then
        sed -i '/head_first=False/d' "${fpath}"
        python3 -c "
import re
t = open('${fpath}').read()
t = re.sub(r',(\s*\))', r'\1', t)
open('${fpath}', 'w').write(t)
" 2>/dev/null || true
        ok "  Fixed: ${fla_wrapper} (removed head_first)"
    fi
done

# ── 7b. Install STP mixer into Zoology ────────────────────────────────────────
echo "  Installing STP mixer into Zoology..."
if [ -f "${REPO_DIR}/mixers/stp_zoology.py" ]; then
    cp "${REPO_DIR}/mixers/stp_zoology.py" "${ZOOLOGY_DIR}/zoology/mixers/stp.py"
    ok "STP mixer → zoology/mixers/stp.py"
else
    fail "mixers/stp_zoology.py not found in ${REPO_DIR}"
    exit 1
fi

# ── 8. Register mixer directory on sys.path ──────────────────────────────────
echo "[8/10] Registering mixers on sys.path..."
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || \
           python3 -c "import site; print(site.getusersitepackages())")
PTH_FILE="${SITE_PKG}/stp_t_p1.pth"
if echo "${MIXER_DIR}" > "${PTH_FILE}" 2>/dev/null; then
    ok "Registered: ${MIXER_DIR}"
    ok "  → ${PTH_FILE}"
else
    USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
    mkdir -p "${USER_SITE}"
    echo "${MIXER_DIR}" > "${USER_SITE}/stp_t_p1.pth"
    ok "Registered (user): ${MIXER_DIR}"
    ok "  → ${USER_SITE}/stp_t_p1.pth"
fi

# ── 9. Install stp_train.py ──────────────────────────────────────────────────
echo "[9/10] Installing stp_train.py..."
if [ -f "${REPO_DIR}/scripts/stp_train.py" ]; then
    if [ ! -f "${ZOOLOGY_DIR}/zoology/train.py.orig" ]; then
        cp "${ZOOLOGY_DIR}/zoology/train.py" "${ZOOLOGY_DIR}/zoology/train.py.orig"
    fi
    cp "${REPO_DIR}/scripts/stp_train.py" "${ZOOLOGY_DIR}/zoology/train.py"
    ok "stp_train.py installed"
else
    warn "stp_train.py not found — WandB-only results"
fi

mkdir -p "${WORKSPACE}/results/runs"
mkdir -p "${WORKSPACE}/checkpoints"
ok "Directories: ${WORKSPACE}/results/runs/, ${WORKSPACE}/checkpoints/"

# ── 10. Verification ─────────────────────────────────────────────────────────
echo "[10/10] Verification..."
WORKSPACE_FOR_PY="${WORKSPACE}" python3 << 'PYEOF'
import sys, os, torch

errors = []
ws = os.environ.get("WORKSPACE_FOR_PY", "")

if not torch.cuda.is_available():
    errors.append("CUDA not available")
else:
    print(f"  ✓ PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")

try:
    from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
    lc = LoggerConfig()
    assert lc.project_name is None and lc.entity is None
    print("  ✓ zoology.config (pydantic v2 OK)")
except Exception as e:
    errors.append(f"zoology.config: {e}")

try:
    from zoology.data.multiquery_ar import MQARConfig
    _ = MQARConfig(vocab_size=8192, input_seq_len=64, num_examples=10, num_kv_pairs=4)
    print("  ✓ zoology.data.multiquery_ar.MQARConfig")
except Exception as e:
    errors.append(f"zoology.data.multiquery_ar: {e}")

try:
    from zoology.mixers.base_conv import BaseConv
    print("  ✓ zoology.mixers.base_conv.BaseConv")
except Exception as e:
    errors.append(f"zoology.mixers.base_conv: {e}")

try:
    from zoology.mixers.attention import MHA
    from zoology.mixers.based import Based
    print("  ✓ zoology.mixers.attention.MHA, based.Based")
except Exception as e:
    errors.append(f"zoology mixers: {e}")

try:
    from fla.layers import GatedLinearAttention, MultiScaleRetention
    print("  ✓ fla.layers (GatedLinearAttention, MultiScaleRetention)")
except Exception as e:
    errors.append(f"fla.layers: {e}")

try:
    from fla_wrappers import RetNetWrapper, GLAWrapper
    print("  ✓ fla_wrappers (RetNetWrapper, GLAWrapper)")
except Exception as e:
    errors.append(f"fla_wrappers: {e}")

try:
    from zoology.mixers.stp import STPAttention, STPAttentionV4
    print("  ✓ stp (STPAttention, STPAttentionV4)")
except Exception as e:
    errors.append(f"stp_v3: {e}")

if not errors and torch.cuda.is_available():
    try:
        from zoology.mixers.stp import STPAttention, STPAttentionV4
        device = "cuda"
        B, T, D = 2, 32, 64
        x = torch.randn(B, T, D).to(device)
        for cls, name in [(STPAttention, "STPv3"), (STPAttentionV4, "STPv4")]:
            m = cls(d_model=D, num_heads=2).to(device)
            y = m(x)
            assert y.shape == (B, T, D), f"Shape mismatch: {y.shape}"
            del m
        print(f"  ✓ STP GPU forward passes (B={B}, T={T}, D={D})")
    except Exception as e:
        errors.append(f"STP forward pass: {e}")

try:
    from zoology.model import LanguageModel
    from zoology.config import ModelConfig, ModuleConfig
    mixer = ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [
            {"name": "zoology.mixers.base_conv.BaseConv",
             "kwargs": {"l_max": 64, "kernel_size": 3, "implicit_long_conv": True}},
            {"name": "zoology.mixers.attention.MHA", "kwargs": {}},
        ]}
    )
    m = LanguageModel(ModelConfig(
        vocab_size=8192, d_model=64, n_layers=2,
        max_position_embeddings=0, sequence_mixer=mixer,
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
    )).cuda()
    out = m(torch.randint(0, 8192, (2, 64)).cuda())
    assert out.shape == (2, 64, 8192), f"Logits shape {out.shape} != (2, 64, 8192)"
    del m
    print("  ✓ Logits shape correct (return_embeddings=False working)")
except Exception as e:
    errors.append(f"return_embeddings check: {e}")

try:
    from zoology.train import Trainer
    if hasattr(Trainer, '_save_checkpoint') and hasattr(Trainer, '_save_run_results'):
        print("  ✓ stp_train.py installed (checkpoint + JSON results)")
    else:
        print("  ⚠ stp_train.py NOT installed — WandB-only results")
except Exception as e:
    errors.append(f"train.py import: {e}")

for d in [f"{ws}/results/runs", f"{ws}/checkpoints"]:
    if d and os.path.isdir(d):
        print(f"  ✓ {d}/")
    elif d:
        print(f"  ⚠ {d}/ missing")

if errors:
    print(f"\n  ERRORS ({len(errors)}):")
    for e in errors:
        print(f"    ✗ {e}")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED")
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " SETUP COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo " Quick start:"
if [ "${IS_PURDUE}" = "1" ]; then
echo "   screen -S stp   # or: tmux new -s stp (if tmux available)"
else
echo "   tmux new -s stp"
fi
echo "   cd ${ZOOLOGY_DIR}"
echo "   export WANDB_MODE=offline"
echo "   export RUN_CONFIG=${REPO_DIR}/configs/run_configs.csv"
echo "   STP_MODELS=stp_v3,stp_v4 python3 -m zoology.launch ${REPO_DIR}/configs/mqar_p1.py"
echo ""
echo " See RUNSHEET.md for full instructions."
echo "════════════════════════════════════════════════════════════════"