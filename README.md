# STP-T MQAR Phase 1

**Evaluating STP-Transformer attention on Multi-Query Associative Recall**

This repo benchmarks STP-T v3 (per-element static decay) and STP-T v4
(per-element content-adaptive decay) against published sub-quadratic
attention baselines using the [Zoology](https://github.com/HazyResearch/zoology)
framework (Arora et al., ICLR 2024).

## Principle

**Zero modifications to the benchmark framework.** All baselines use Zoology's
built-in model factories and published MQAR data configuration. We only add
our two STP-T sequence mixers. This ensures fair, reproducible comparison.

**Consistency guarantee:** Every model uses the same Hybrid(BaseConv + mixer)
wrapper, num_heads=2, max_position_embeddings=0, state_mixer=Identity, and
block_type=TransformerBlock (except Mamba2 which uses Mamba2Block with its own conv).

## Models

| Model | Type | Decay Granularity | Wrapper | Source |
|-------|------|-------------------|---------|--------|
| Attention (MHA) | O(T²) | N/A (full softmax) | Hybrid(conv+MHA) | Zoology built-in |
| Based | Sub-quadratic | Feature map | Hybrid(conv+Based) | Zoology built-in |
| RetNet | Sub-quadratic | Scalar per-head | Hybrid(conv+RetNet) | fla (Sun et al. 2023) |
| GLA | Sub-quadratic | Per-row adaptive | Hybrid(conv+GLA) | fla (Yang et al. 2023) |
| DeltaNet | Sub-quadratic | Delta rule | Hybrid(conv+DeltaNet) | fla (Yang et al. 2024) |
| Gated DeltaNet | Sub-quadratic | Gated delta rule | Hybrid(conv+GatedDeltaNet) | fla (Yang et al. 2024) |
| Mamba2 | Sub-quadratic | Selective SSM | Mamba2Block (own conv) | fla (Gu & Dao 2024) |
| **STP-T v3** | **Sub-quadratic** | **Per-element static** | **Hybrid(conv+STP)** | **This work** |
| **STP-T v4** | **Sub-quadratic** | **Per-element adaptive** | **Hybrid(conv+STP)** | **This work** |

## Quick Start

```bash
# 1. Clone
cd /workspace
git clone https://github.com/raisul1212/stp-t-mqar-p1.git

# 2. Setup (installs Zoology + deps, ~10 min)
bash stp-t-mqar-p1/scripts/setup.sh

# 3. Run ALL models
tmux new -s stp
cd /workspace/zoology
export WANDB_MODE=offline
python3 -m zoology.launch /workspace/stp-t-mqar-p1/configs/mqar_p1.py 2>&1 | tee /workspace/experiment_log.txt

# 3b. Run SPECIFIC models (use STP_MODELS env var)
STP_MODELS=stp_v3,stp_v4 python3 -m zoology.launch /workspace/stp-t-mqar-p1/configs/mqar_p1.py
STP_MODELS=attention,retnet,gla python3 -m zoology.launch /workspace/stp-t-mqar-p1/configs/mqar_p1.py
```

Available model names: `attention`, `based`, `mamba2`, `delta_net`, `gated_delta_net`, `gla`, `retnet`, `stp_v3`, `stp_v4`

## Results

Results are logged via WandB (offline mode → local `wandb/` directory).

Key metrics per run:
- `valid/accuracy` — overall validation accuracy
- `valid/num_kv_pairs/accuracy-{N}` — accuracy per difficulty level (N = 4,8,16,32,64,128,256)
- `train/loss` — training loss

Save results:
```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
tar czf /workspace/stp_mqar_p1_${TIMESTAMP}.tar.gz \
    /workspace/experiment_log.txt \
    /workspace/zoology/wandb/ \
    2>/dev/null
```

## Repo Structure

```
stp-t-mqar-p1/
├── configs/
│   └── mqar_p1.py          # Main benchmark config (uses Zoology factories)
├── mixers/
│   └── stp_zoology.py      # STP-T v3 and v4 sequence mixers (only custom code)
├── scripts/
│   └── setup.sh            # One-shot environment setup
├── README.md
├── RUNSHEET.md              # Step-by-step deployment guide
└── .gitignore
```

## What We Claim

1. **STP-T v3 ≥ RetNet/Based**: Per-element static decay should outperform
   scalar per-head decay and fixed feature maps on recall tasks.

2. **STP-T v4 ≈ GLA with fewer parameters**: Content-adaptive per-element
   decay should match or exceed per-row adaptive decay (GLA) while using
   physics-informed parameter sharing (256× fewer decay parameters).

## References

- Arora et al. "Zoology: Measuring and Improving Recall in Efficient Language Models" (ICLR 2024)
- Yang et al. "Gated Linear Attention Transformers with Hardware-Efficient Training" (2023)
- Yang et al. "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (2024)
- Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2024)
- Sun et al. "Retentive Network: A Successor to Transformer for Large Language Models" (2023)
