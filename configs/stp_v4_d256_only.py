"""
STP-T v4 d256 Only: Targeted config for missing v4 d_model=256 runs.

This is a minimal extract from mqar_p1.py — same data, same model structure,
same hyperparameters. Only generates the 4 LR runs for stp_v4 at d_model=256.

Usage:
  cd /workspace/zoology
  python -m zoology.launch /workspace/stp-t-mqar-p1/configs/stp_v4_d256_only.py
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "stp-v4-d256-" + sweep_id
VOCAB_SIZE = 8_192

# ── Data (identical to mqar_p1.py) ──

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

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(256, 256 // 8),
    cache_dir="/workspace/zoology_cache",
)

# ── Model (stp_v4 at d_model=256 only) ──

d_model = 256

conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    },
)

stp_mixer = dict(
    name="zoology.mixers.stp.STPAttentionV4",
    kwargs={
        "num_heads": 2,
        "chunk_size": 64,
        "use_sign": True,
        "gamma_init": 0.1,
        "W_LTM_init_std": 0.5,
    },
)

hybrid_mixer = ModuleConfig(
    name="zoology.mixers.hybrid.Hybrid",
    kwargs={"configs": [conv_mixer, stp_mixer]},
)

model = ModelConfig(
    block_type="TransformerBlock",
    name="stp_v4",
    d_model=d_model,
    n_layers=2,
    max_position_embeddings=0,
    vocab_size=VOCAB_SIZE,
    sequence_mixer=hybrid_mixer,
    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
)

# ── Sweep (4 LRs, identical to mqar_p1.py) ──

configs = []
for lr in np.logspace(-3, -1.5, 4):
    run_id = f"stp_v4-d{d_model}-lr{lr:.1e}"
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

print(f"Generated {len(configs)} configs:")
for c in configs:
    print(f"  {c.run_id}")
