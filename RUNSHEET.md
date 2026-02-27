# RUNSHEET: STP-T MQAR Phase 1

Step-by-step deployment with automated results saving. Updated Feb 27, 2026.

## Requirements

- 1× GPU with ≥24GB VRAM (A100, RTX 4090, RTX 5090, etc.)
- Linux with CUDA drivers
- ~10GB disk for packages + cache
- ~24-48 hours for full sweep, ~2-4 hours for a single model group

## Steps

### 1. Start GPU instance

Any cloud provider (RunPod, Lambda, Vast.ai) with a PyTorch template.
Verify with `nvidia-smi`.

### 2. Clone this repo

```bash
cd /workspace
git clone https://github.com/raisul1212/stp-t-mqar-p1.git
```

### 3. Run setup

```bash
bash stp-t-mqar-p1/scripts/setup.sh
```

This does everything automatically:
- Installs Zoology, flash-linear-attention, pyyaml, and other deps
- Applies the `return_embeddings=False` fix to Zoology's `model.py`
- Copies STP mixer into Zoology's mixers directory
- **Patches `train.py`** with checkpoint saving + JSON results + best-epoch tracking
- Copies `extract_results.py` to `/workspace/`
- Creates `/workspace/results/` and `/workspace/checkpoints/` directories
- Runs full verification (GPU, imports, forward pass, logits shape, patch check)

If setup prints `ALL CHECKS PASSED`, proceed. If not, fix the reported error.

### 4. Launch benchmark

Open a tmux session (survives SSH disconnects):

```bash
tmux new -s stp
```

#### Option A: Fully automated (recommended)

```bash
# Run only STP models (~2-4h)
bash stp-t-mqar-p1/scripts/run.sh stp_v3,stp_v4

# STP vs direct competitors (~6-8h)
bash stp-t-mqar-p1/scripts/run.sh stp_v3,stp_v4,retnet,gla

# Everything (~24-48h)
bash stp-t-mqar-p1/scripts/run.sh
```

`run.sh` does three things in sequence:
1. Launches the sweep via `zoology.launch`
2. Runs `extract_results.py` to aggregate all results
3. Runs `archive.sh` to create a downloadable tarball

#### Option B: Manual (step by step)

```bash
cd /workspace/zoology
export WANDB_MODE=offline
export STP_RESULTS_DIR=/workspace/results
export STP_CHECKPOINT_DIR=/workspace/checkpoints
export STP_SAVE_CHECKPOINTS=best

# Pick your models
STP_MODELS=stp_v3,stp_v4 python3 -m zoology.launch /workspace/stp-t-mqar-p1/configs/mqar_p1.py 2>&1 | tee /workspace/experiment_log.txt
```

Available model names: `attention`, `based`, `mamba2`, `delta_net`, `gated_delta_net`, `gla`, `retnet`, `stp_v3`, `stp_v4`

Detach tmux: `Ctrl+B`, then `D`
Reconnect: `tmux attach -t stp`

### 5. Monitor

```bash
# Progress
tail -f /workspace/experiment_log.txt

# GPU utilization
nvidia-smi -l 10

# Completed runs (each run writes a JSON when done)
ls /workspace/results/runs/ | wc -l

# Quick look at a completed run
cat /workspace/results/runs/stp_v3-d64-lr1.0e-03.json | python3 -m json.tool
```

### 6. Extract results (if you used Option B)

If you used `run.sh` (Option A), this already ran automatically. Otherwise:

```bash
python3 /workspace/extract_results.py
```

This reads from three sources (in priority order):
1. Per-run JSONs in `/workspace/results/runs/` (most complete)
2. WandB offline dirs in `/workspace/zoology/wandb/`
3. Experiment log at `/workspace/experiment_log.txt`

Outputs:

```
/workspace/results/
├── runs/                     # Per-run JSONs (written during training)
│   ├── stp_v3-d64-lr1.0e-03.json
│   ├── attention-d128-lr3.2e-03.json
│   └── ...
├── summary.json              # All runs merged, full detail
├── summary.csv               # Spreadsheet-friendly, one row per run
├── best_per_model.json       # Best LR per (model, d_model) — cite this in the paper
└── comparison_table.txt      # ASCII table printed to terminal
```

### 7. Archive before stopping the pod

If you used `run.sh`, the archive was already created. Otherwise:

```bash
bash stp-t-mqar-p1/scripts/archive.sh
```

This creates `/workspace/stp_mqar_p1_YYYYMMDD_HHMM.tar.gz` containing:
- `results/` (per-run JSONs + aggregated summaries)
- `checkpoints/` (model weights)
- `experiment_log.txt`
- `zoology/wandb/` (WandB offline data)
- `stp-t-mqar-p1/` (the repo itself)

Download before stopping:
```bash
# From your local machine
scp -P <port> root@<pod-ip>:/workspace/stp_mqar_p1_*.tar.gz .

# Or use runpodctl
runpodctl send /workspace/stp_mqar_p1_*.tar.gz
```

Then stop the pod.

## What gets saved per run

The patched `train.py` writes three things automatically after each run:

**1. Per-run JSON** → `/workspace/results/runs/{run_id}.json`
```
Contains: model_name, d_model, num_parameters, learning_rate,
          best_epoch, best_valid_accuracy, best_accuracy_by_kv_pairs,
          final_epoch, final_valid_accuracy, final_accuracy_by_kv_pairs,
          epoch_history, checkpoint_path, training_time_seconds
```

**2. Model checkpoints** → `/workspace/checkpoints/{run_id}/`
```
best.pt   — saved whenever validation accuracy improves
final.pt  — saved at end of training
```
Each `.pt` contains: `model_state_dict`, `optimizer_state_dict`, `metrics`, `epoch`

Control with `STP_SAVE_CHECKPOINTS`:
- `best` (default) — best.pt + final.pt
- `all` — also epoch0.pt, epoch1.pt, ... (large disk usage)
- `none` — skip checkpoints entirely

**3. WandB logs** → `/workspace/zoology/wandb/offline-run-*/`
Standard WandB offline data (same as before the patch).

## Environment variables

| Variable | Default | Options | Purpose |
|----------|---------|---------|---------|
| `STP_MODELS` | `all` | comma-separated model names | Which models to run |
| `STP_SAVE_CHECKPOINTS` | `best` | `best`, `all`, `none` | Checkpoint saving mode |
| `STP_RESULTS_DIR` | `/workspace/results` | any path | Per-run JSONs + aggregated results |
| `STP_CHECKPOINT_DIR` | `/workspace/checkpoints` | any path | Model .pt files |
| `WANDB_MODE` | (unset) | `offline` | Must be `offline` on RunPod |

## Reloading a checkpoint

```python
import torch
from zoology.config import ModelConfig, ModuleConfig
from zoology.model import LanguageModel

# Rebuild architecture (must match training config)
model = LanguageModel(ModelConfig(
    vocab_size=8192, d_model=128, n_layers=2,
    max_position_embeddings=0,
    sequence_mixer=ModuleConfig(
        name="zoology.mixers.stp.STPAttention",
        kwargs={"num_heads": 2, "chunk_size": 64, "use_sign": True,
                "gamma_init": 0.1, "W_LTM_init_std": 0.5}
    ),
    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
))

ckpt = torch.load("/workspace/checkpoints/stp_v3-d128-lr3.2e-03/best.pt")
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded epoch {ckpt['epoch']}, accuracy {ckpt['best_accuracy']:.4f}")

# Inspect STP physics parameters
mixer = model.backbone.layers[1].mixer.modules_list[1]
print(f"W_LTM: mean={mixer.W_LTM.mean():.4f}, std={mixer.W_LTM.std():.4f}")
print(f"V_gs: {mixer.V_gs.data}")
print(f"gamma: {mixer.gamma.data}")
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named zoology` | `cd /workspace/zoology && pip install -e . --break-system-packages --no-deps` |
| `No module named fla` | `pip install flash-linear-attention --break-system-packages --no-deps` |
| `got multiple values for d_model` | STP mixer missing `layer_idx` param — rerun setup.sh |
| `No space left on device` | `pip cache purge && rm -rf /root/.cache/pip` |
| All accuracies ~25% | Check `grep return_embeddings /workspace/zoology/zoology/model.py` — must show `False` |
| OOM error | Reduce batch_size in mqar_p1.py: change 256 to 128 |
| `STP_MODELS filter: ... → 0 models` | Typo in model name — check available names above |
| `train.py NOT patched` in setup output | Rerun `setup.sh`, or manually: `cp stp-t-mqar-p1/scripts/stp_train.py /workspace/zoology/zoology/train.py` |
| No per-run JSONs after sweep | Check env: `echo $STP_RESULTS_DIR` (must be set before launch) |
| Checkpoints too large | Set `STP_SAVE_CHECKPOINTS=none` or `STP_SAVE_CHECKPOINTS=best` |
