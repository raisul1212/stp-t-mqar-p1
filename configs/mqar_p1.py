"""
STP-T MQAR Phase 1: CSV-Driven Benchmark Config (v2)
=====================================================

Updated from hardcoded config to CSV-driven (ported from P2 pattern).
Mixer classes unchanged: zoology.mixers.stp.STPAttention (v3),
zoology.mixers.stp.STPAttentionV4 (v4).

CSV path: RUN_CONFIG env var or ./run_configs.csv
Model filter: STP_MODELS=stp_v3,retnet
"""

import csv, os, sys, uuid
from pathlib import Path
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

_HERE = Path(__file__).parent
_CSV_PATH = Path(os.environ.get("RUN_CONFIG", str(_HERE / "run_configs.csv")))
if not _CSV_PATH.exists():
    print(f"[mqar_p1] ERROR: {_CSV_PATH} not found", file=sys.stderr); sys.exit(1)

_ACTIVE = set(os.environ.get("STP_MODELS", "").split(",")) - {""}
def _include(name): return (not _ACTIVE) or (name in _ACTIVE)

def _load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            if row.get("enabled", "1").strip() != "1": continue
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows

_ALL_ROWS = _load_csv(_CSV_PATH)
print(f"[mqar_p1] {len(_ALL_ROWS)} enabled rows from {_CSV_PATH.name}", file=sys.stderr)

VOCAB_SIZE = 8192
SWEEP_NAME = "stp-t-mqar-p1-" + uuid.uuid4().hex[:6]
SEQ_LEN_MAX = 1024

_TRAIN = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
_TEST = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128,  num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

def _data(bs): return DataConfig(train_configs=_TRAIN, test_configs=_TEST,
    batch_size=(bs, max(1, bs // 8)),
    cache_dir=os.path.join(os.environ.get("WORKSPACE", os.path.expanduser("~")), "zoology_cache"))

def _conv(): return {"name": "zoology.mixers.base_conv.BaseConv",
    "kwargs": {"l_max": SEQ_LEN_MAX, "kernel_size": 3, "implicit_long_conv": True}}

def _hybrid(seq): return ModuleConfig(name="zoology.mixers.hybrid.Hybrid",
    kwargs={"configs": [_conv(), seq]})

def _seq(row):
    m, nh = row["model"], int(row["num_heads"])
    if m == "attention":
        return {"name": "zoology.mixers.attention.MHA", "kwargs": {"num_heads": nh}}
    elif m == "based":
        return {"name": "zoology.mixers.based.Based", "kwargs": {
            "l_max": SEQ_LEN_MAX, "feature_dim": 16, "feature_name": "taylor_exp",
            "num_key_value_heads": 1, "num_heads": 1, "train_view": "quadratic"}}
    elif m == "retnet":
        return {"name": "fla_wrappers.RetNetWrapper",
                "kwargs": {"num_heads": nh, "mode": "fused_chunk"}}
    elif m == "gla":
        return {"name": "fla_wrappers.GLAWrapper", "kwargs": {
            "num_heads": nh, "mode": "chunk", "expand_k": 0.5,
            "expand_v": 1.0, "use_output_gate": True}}
    elif m == "gated_delta_net":
        return {"name": "zoology.mixers.gated_delta_net.GatedDeltaNet",
                "kwargs": {"num_heads": nh}}
    elif m == "stp_v3":
        return {"name": "zoology.mixers.stp.STPAttention", "kwargs": {
            "num_heads": nh, "chunk_size": 64, "use_sign": True,
            "gamma_init": 0.1, "W_LTM_init_std": 0.5}}
    elif m == "stp_v4":
        return {"name": "zoology.mixers.stp.STPAttentionV4", "kwargs": {
            "num_heads": nh, "chunk_size": 64, "use_sign": True,
            "gamma_init": 0.1, "W_LTM_init_std": 0.5}}
    else:
        raise ValueError(f"Unknown model '{m}'")

configs = []
for row in _ALL_ROWS:
    if not _include(row.get("model", "")): continue
    try:
        configs.append(TrainConfig(
            model=ModelConfig(d_model=int(row["d_model"]), n_layers=int(row["n_layers"]),
                max_position_embeddings=0, vocab_size=VOCAB_SIZE,
                sequence_mixer=_hybrid(_seq(row)),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={})),
            data=_data(int(row["batch_size"])),
            learning_rate=float(row["lr"]), max_epochs=int(row["max_epochs"]),
            run_id=row["run_id"], sweep_id=SWEEP_NAME,
            slice_keys=["num_kv_pairs"], logger=LoggerConfig()))
    except Exception as e:
        print(f"[mqar_p1] ERROR: {row.get('run_id','?')}: {e}", file=sys.stderr); raise

print(f"[mqar_p1] {len(configs)} configs, sweep={SWEEP_NAME}", file=sys.stderr)
if configs:
    ms = sorted(set(c.run_id.split("-")[0] for c in configs))
    print(f"[mqar_p1] Models: {', '.join(ms)}", file=sys.stderr)
