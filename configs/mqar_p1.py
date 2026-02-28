"""
STP-T MQAR Phase 1: Sub-Quadratic Attention Benchmarks
========================================================

Uses Zoology's EXACT published MQAR benchmark config structure.
All baselines use Zoology's built-in model factories.
Only STP-T v3 and v4 are added as new models.

CONSISTENCY RULES (all models must satisfy):
  - Wrapped in Hybrid(BaseConv + mixer) OR use own block_type (Mamba2Block)
  - num_heads = 2 (matching all Zoology factory baselines)
  - max_position_embeddings = 0 (no positional encoding, matching factories)
  - state_mixer = Identity (no FFN, matching published config)
  - block_type = TransformerBlock (except Mamba2 which uses Mamba2Block)

Models:
  Baselines (from Zoology / fla):
    1. Attention (MHA)       — O(T²) upper bound, Hybrid(conv+MHA)
    2. Based                 — linear attention + feature maps, Hybrid(conv+Based)
    3. Mamba2                — selective SSM, Mamba2Block (own conv)
    4. DeltaNet              — delta rule, Hybrid(conv+DeltaNet)
    5. Gated DeltaNet        — gated delta, Hybrid(conv+GatedDeltaNet)
    6. GLA                   — per-row adaptive decay, Hybrid(conv+GLA)
    7. RetNet                — scalar per-head decay, Hybrid(conv+RetNet)

  This Work:
    8. STP-T v3              — per-element STATIC decay, Hybrid(conv+STP)
    9. STP-T v4              — per-element ADAPTIVE decay, Hybrid(conv+STP)

Data (identical to Zoology published config):
  - vocab_size = 8192
  - Train: 100K examples (4-pair), 20K each (8,16,32,64-pair)
  - Test: 1K examples each at 4,8,16,32,64,128,256-pair
  - Varying input_seq_len per difficulty

Usage:
  cd /workspace/zoology
  python -m zoology.launch /workspace/stp-t-mqar-p1/configs/mqar_p1.py
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "stp-t-mqar-p1-" + sweep_id

VOCAB_SIZE = 8_192

# ─────────────────────────────────────────────────────────────
# DATA CONFIG — identical to Zoology published config
# ─────────────────────────────────────────────────────────────

train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]

test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128,  num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

input_seq_len = max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, batch_size // 8),
    cache_dir="/workspace/zoology_cache",
)

# ─────────────────────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────────────────────

models = []

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# Short convolution — part of the standard Zoology recipe.
# All TransformerBlock-based models get this via Hybrid wrapper.
# Mamba2 has its own internal conv (uses Mamba2Block).
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)


# ── Baselines from Zoology's models_repo ──
# These factory functions handle their own d_model sweeps and kwargs.
# Each wraps its mixer in Hybrid(BaseConv + mixer) automatically.
from zoology.experiments.models_repo import (
    add_attention,
    add_based,
    add_mamba2,
    add_delta_net,
    add_gla,
    add_gated_delta_net,
)

models = add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_based(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_mamba2(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)

# Filter: keep all factory baselines
included = [
    "attention",
    "based",
    "mamba2",
    "delta_net",
    "gla",
    "gated_delta_net",
]
models = [m for m in models if any([i in m.name for i in included])]


# ── RetNet (from fla, not in Zoology's models_repo) ──
# RetNet uses scalar per-head decay — direct v3 comparison target.
# CONSISTENCY: Wrapped in Hybrid(conv+RetNet), num_heads=2,
#              max_position_embeddings=0, block_type=TransformerBlock.
try:
    from fla.layers import MultiScaleRetention as _
    for d_model in [64, 128, 256]:
        retnet_mixer = dict(
            name="fla.layers.MultiScaleRetention",
            kwargs={"num_heads": 2, "mode": "fused_chunk"},
        )
        # Wrap in Hybrid(BaseConv + RetNet) to match all other baselines
        hybrid_mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, retnet_mixer]},
        )
        models.append(
            ModelConfig(
                block_type="TransformerBlock",
                name=f"retnet",
                d_model=d_model,
                n_layers=2,
                max_position_embeddings=0,   # Match baselines: no pos encoding
                vocab_size=VOCAB_SIZE,
                sequence_mixer=hybrid_mixer,
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            )
        )
except ImportError:
    print("WARNING: Skipping RetNet — fla.layers.MultiScaleRetention not available")


# ── STP-T v3 and v4 (THIS WORK) ──
# CONSISTENCY: Wrapped in Hybrid(conv+STP), num_heads=2,
#              max_position_embeddings=0, block_type=TransformerBlock.
# The ONLY difference from baselines is the sequence mixer module.

for d_model in [64, 128, 256]:
    for model_name, mixer_class in [
        ("stp_v3", "zoology.mixers.stp.STPAttention"),
        ("stp_v4", "zoology.mixers.stp.STPAttentionV4"),
    ]:
        stp_mixer = dict(
            name=mixer_class,
            kwargs={
                "num_heads": 2,            # Match all baselines
                "chunk_size": 64,
                "use_sign": True,
                "gamma_init": 0.1,
                "W_LTM_init_std": 0.5,
            },
        )
        # Wrap in Hybrid(BaseConv + STP) to match all other baselines
        hybrid_mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, stp_mixer]},
        )
        models.append(
            ModelConfig(
                block_type="TransformerBlock",
                name=f"{model_name}",
                d_model=d_model,
                n_layers=2,
                max_position_embeddings=0,   # Match baselines: no pos encoding
                vocab_size=VOCAB_SIZE,
                sequence_mixer=hybrid_mixer,
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            )
        )


# ─────────────────────────────────────────────────────────────
# RUNTIME MODEL FILTER
# ─────────────────────────────────────────────────────────────
# Use STP_MODELS env var to select which models to run:
#   STP_MODELS=stp_v3,stp_v4,attention  → only those 3
#   STP_MODELS=all                       → run everything (default)
#
# Examples:
#   STP_MODELS=stp_v3,stp_v4 python -m zoology.launch configs/mqar_p1.py
#   STP_MODELS=attention,retnet python -m zoology.launch configs/mqar_p1.py

import os
_model_filter = os.environ.get("STP_MODELS", "all").strip().lower()
if _model_filter != "all":
    _selected = [s.strip() for s in _model_filter.split(",") if s.strip()]
    _before = len(models)
    models = [m for m in models if any(s in m.name for s in _selected)]
    print(f"STP_MODELS filter: {_selected} → {_before} → {len(models)} models")


# ─────────────────────────────────────────────────────────────
# TRAINING CONFIGS — match published Zoology setup
# ─────────────────────────────────────────────────────────────

configs = []

for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-d{model.d_model}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
        )
        configs.append(config)

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

print(f"Sweep: {sweep_name}")
print(f"Total models: {len(models)}")
for m in models:
    mixer_info = ""
    if hasattr(m.sequence_mixer, 'name'):
        mixer_info = f" mixer={m.sequence_mixer.name}"
    print(f"  - {m.name} (d_model={m.d_model}, pos_emb={m.max_position_embeddings}{mixer_info})")
print(f"LR sweep: {[f'{lr:.1e}' for lr in np.logspace(-3, -1.5, 4)]}")
print(f"Total configs (models x LRs): {len(configs)}")
