"""
STP-T v4 d256 Only: Targeted config for missing v4 d_model=256 runs.

Usage:
  cd /workspace/zoology
  python -m zoology.launch /workspace/stp-t-mqar-p1/configs/stp_v4_d256_only.py

This generates only 4 runs (one per LR) for stp_v4 at d_model=256.
All other settings match mqar_p1.py exactly.
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
VOCAB_SIZE = 8_192
d_model = 256
num_heads = 2

train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=2048, num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=4096, num_examples=1_000, num_kv_pairs=256),
]

configs = []
for lr in np.logspace(-3, -1.5, 4):
    run_id = f"stp_v4-d{d_model}-lr{lr:.1e}"
    model = ModelConfig(
        name="stp_v4",
        d_model=d_model,
        n_layers=4,
        max_position_embeddings=0,
        vocab_size=VOCAB_SIZE,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid",
            kwargs=dict(configs=[
                ModuleConfig(name="zoology.mixers.base_conv", kwargs=dict(kernel_size=3, implicit_long_conv=True)),
                ModuleConfig(name="zoology.mixers.stp.STPAttentionV4", kwargs=dict(num_heads=num_heads)),
            ])
        ),
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        block_type="TransformerBlock",
    )
    config = TrainConfig(
        data=DataConfig(
            vocab_size=VOCAB_SIZE,
            cache_dir="/workspace/zoology_cache",
            train_configs=train_configs,
            test_configs=test_configs,
            batch_size=256,
            num_workers=4,
        ),
        model=model,
        logger=LoggerConfig(),
        learning_rate=float(lr),
        max_epochs=32,
        slice_keys=["num_kv_pairs"],
        run_id=run_id,
        sweep_id=sweep_id,
    )
    configs.append(config)

print(f"Generated {len(configs)} configs:")
for c in configs:
    print(f"  {c.run_id}")
