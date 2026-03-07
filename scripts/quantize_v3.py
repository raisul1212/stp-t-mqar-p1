#!/usr/bin/env python3
"""
STP-T v3 Post-Training Quantization for RRAM CIM
=================================================

Loads a trained Zoology checkpoint, quantizes static components to
N bits, evaluates on MQAR. No training — pure evaluation.

Usage:
  python3 quantize_v3.py \
      --checkpoint ~/checkpoints/stp_v3-d128-lr3.0e-03-quant/best.pt \
      --d_model 128 --bits 1 2 3 4 5 6 8 \
      --output ~/results/quant_d128.json

Author: RISE Lab, Purdue University
Date: March 2026
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# Ensure zoology is importable regardless of working directory.
# Zoology may be pip-installed or just cloned at ~/zoology.
_ZOOLOGY_DIR = os.environ.get("ZOOLOGY_DIR", os.path.expanduser("~/zoology"))
if _ZOOLOGY_DIR not in sys.path and os.path.isdir(_ZOOLOGY_DIR):
    sys.path.insert(0, _ZOOLOGY_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. QUANTIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def uniform_quantize(tensor, n_bits, symmetric=True):
    """Symmetric uniform quantization to n_bits."""
    if n_bits >= 16:
        return tensor
    n_levels = 2 ** n_bits
    if symmetric:
        abs_max = tensor.abs().max().clamp(min=1e-8)
        scale = abs_max / (n_levels // 2 - 1)
        quantized = torch.round(tensor / scale).clamp(-(n_levels // 2), n_levels // 2 - 1)
        return quantized * scale
    else:
        t_min, t_max = tensor.min(), tensor.max()
        scale = (t_max - t_min).clamp(min=1e-8) / (n_levels - 1)
        zero_point = torch.round(-t_min / scale).clamp(0, n_levels - 1)
        quantized = torch.round(tensor / scale + zero_point).clamp(0, n_levels - 1)
        return (quantized - zero_point) * scale


# ═══════════════════════════════════════════════════════════════
# 2. LOAD ZOOLOGY CHECKPOINT
# ═══════════════════════════════════════════════════════════════

def load_zoology_model(checkpoint_path, d_model, device="cuda"):
    """Load trained Zoology LanguageModel from checkpoint."""
    from zoology.config import ModelConfig, ModuleConfig
    from zoology.model import LanguageModel

    conv_cfg = {
        "name": "zoology.mixers.base_conv.BaseConv",
        "kwargs": {"l_max": 1024, "kernel_size": 3, "implicit_long_conv": True},
    }
    stp_cfg = {
        "name": "zoology.mixers.stp.STPAttention",
        "kwargs": {
            "num_heads": 2, "chunk_size": 64,
            "use_sign": True, "gamma_init": 0.1, "W_LTM_init_std": 0.5,
        },
    }
    hybrid = ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_cfg, stp_cfg]},
    )
    model = LanguageModel(ModelConfig(
        d_model=d_model, n_layers=2, max_position_embeddings=0,
        vocab_size=8192, sequence_mixer=hybrid,
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
    ))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Accuracy: {ckpt.get('best_accuracy', 0):.4f}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, ckpt


# ═══════════════════════════════════════════════════════════════
# 3. QUANTIZATION STRATEGIES
# ═══════════════════════════════════════════════════════════════

def quantize_wltm_only(model, bits):
    """Quantize only W_LTM in STPAttention layers."""
    qmodel = copy.deepcopy(model)
    for _, module in qmodel.named_modules():
        if type(module).__name__ == "STPAttention":
            with torch.no_grad():
                module.W_LTM.data = uniform_quantize(module.W_LTM.data, bits)
    return qmodel


def quantize_all_stp(model, bits):
    """Quantize all STP physics parameters."""
    qmodel = copy.deepcopy(model)
    stp_params = ["W_LTM", "V_gs", "V_T0", "beta_tau", "beta_gm",
                  "C_ch", "gamma", "alpha_ppd", "IC_threshold"]
    for _, module in qmodel.named_modules():
        if type(module).__name__ == "STPAttention":
            with torch.no_grad():
                for pname in stp_params:
                    if hasattr(module, pname):
                        p = getattr(module, pname)
                        p.data = uniform_quantize(p.data, bits)
    return qmodel


def quantize_full_model(model, bits):
    """Quantize ALL parameters (embeddings, QKV, STP, everything)."""
    qmodel = copy.deepcopy(model)
    with torch.no_grad():
        for _, param in qmodel.named_parameters():
            param.data = uniform_quantize(param.data, bits)
    return qmodel


def quantize_single_param(model, param_name, bits):
    """Quantize one specific parameter in STPAttention layers."""
    qmodel = copy.deepcopy(model)
    for _, module in qmodel.named_modules():
        if type(module).__name__ == "STPAttention":
            if hasattr(module, param_name):
                with torch.no_grad():
                    p = getattr(module, param_name)
                    p.data = uniform_quantize(p.data, bits)
    return qmodel


# ═══════════════════════════════════════════════════════════════
# 4. MQAR DATA + EVALUATION
# ═══════════════════════════════════════════════════════════════

def generate_mqar_batch(batch_size, n_kv_pairs, vocab_size=8192, device="cuda"):
    """Generate MQAR batch."""
    seq_len = max(64, 4 * n_kv_pairs)
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    keys = torch.stack([torch.randperm(vocab_size, device=device)[:n_kv_pairs]
                        for _ in range(batch_size)])
    values = torch.randint(0, vocab_size, (batch_size, n_kv_pairs), device=device)

    for i in range(n_kv_pairs):
        input_ids[:, 2 * i] = keys[:, i]
        input_ids[:, 2 * i + 1] = values[:, i]

    query_start = seq_len - n_kv_pairs
    for b in range(batch_size):
        perm = torch.randperm(n_kv_pairs, device=device)
        for i in range(n_kv_pairs):
            qi = perm[i].item()
            input_ids[b, query_start + i] = keys[b, qi]
            targets[b, query_start + i] = values[b, qi]
            mask[b, query_start + i] = True

    return {"input_ids": input_ids, "targets": targets, "mask": mask}


@torch.no_grad()
def evaluate_mqar(model, kv_counts, vocab_size=8192, n_eval=1000,
                  batch_size=128, device="cuda"):
    """Evaluate MQAR accuracy at multiple KV counts."""
    model.eval()
    results = {}
    total_correct = 0
    total_count = 0

    for kv in kv_counts:
        correct = 0
        count = 0
        n_batches = max(1, n_eval // batch_size)

        for _ in range(n_batches):
            bs = min(batch_size, n_eval - count)
            if bs <= 0:
                break
            batch = generate_mqar_batch(bs, kv, vocab_size, device)
            logits = model(batch["input_ids"])
            preds = logits[batch["mask"]].argmax(dim=-1)
            tgts = batch["targets"][batch["mask"]]
            correct += (preds == tgts).sum().item()
            count += tgts.numel()

        acc = 100.0 * correct / max(count, 1)
        results[kv] = acc
        total_correct += correct
        total_count += count

    overall = 100.0 * total_correct / max(total_count, 1)
    return {"overall": overall, "per_kv": results}


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv_counts = [4, 8, 16, 32, 64, 128, 256]

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"d_model: {args.d_model}")
    print(f"Bits: {args.bits}")

    # ─── Load ───
    print(f"\n{'='*60}")
    print("Loading checkpoint...")
    model, ckpt = load_zoology_model(args.checkpoint, args.d_model, device)

    # ─── FP Baseline ───
    print(f"\n{'='*60}")
    print("FP BASELINE")
    print(f"{'='*60}", flush=True)
    fp = evaluate_mqar(model, kv_counts, device=device)
    print(f"  Overall: {fp['overall']:.1f}%")
    for kv in kv_counts:
        print(f"    kv={kv}: {fp['per_kv'][kv]:.1f}%")

    results = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "d_model": args.d_model,
            "bits": args.bits,
            "ckpt_epoch": ckpt.get("epoch"),
            "ckpt_accuracy": ckpt.get("best_accuracy"),
        },
        "fp_baseline": fp,
        "quantized": {},
    }

    def header():
        return (f"{'Bits':>6} {'Cells/wt':>8} {'Overall':>9} {'Δ':>7}  " +
                "  ".join(f"{'kv'+str(kv):>6}" for kv in kv_counts))

    def row(bits, qr, tag=""):
        d = qr["overall"] - fp["overall"]
        cells = f"{2*bits}" if bits < 16 else "FP"
        kvs = "  ".join(f"{qr['per_kv'].get(kv,0):5.1f}%" for kv in kv_counts)
        return f"  {bits:>4}b {cells:>8} {qr['overall']:>8.1f}% {d:>+6.1f}  {kvs}"

    # ─── A. W_LTM only ───
    print(f"\n{'='*60}")
    print("A. W_LTM ONLY (RRAM crossbar weight matrix)")
    print(f"{'='*60}")
    print(header(), flush=True)

    results["quantized"]["wltm_only"] = {}
    for bits in args.bits:
        qm = quantize_wltm_only(model, bits)
        qr = evaluate_mqar(qm, kv_counts, device=device)
        print(row(bits, qr), flush=True)
        results["quantized"]["wltm_only"][bits] = qr
        del qm; torch.cuda.empty_cache() if device == "cuda" else None

    # ─── B. All STP params ───
    print(f"\n{'='*60}")
    print("B. ALL STP PARAMS (W_LTM + physics scalars)")
    print(f"{'='*60}")
    print(header(), flush=True)

    results["quantized"]["all_stp"] = {}
    for bits in args.bits:
        qm = quantize_all_stp(model, bits)
        qr = evaluate_mqar(qm, kv_counts, device=device)
        print(row(bits, qr), flush=True)
        results["quantized"]["all_stp"][bits] = qr
        del qm; torch.cuda.empty_cache() if device == "cuda" else None

    # ─── C. Full model ───
    print(f"\n{'='*60}")
    print("C. FULL MODEL (all parameters)")
    print(f"{'='*60}")
    print(header(), flush=True)

    results["quantized"]["full_model"] = {}
    for bits in args.bits:
        qm = quantize_full_model(model, bits)
        qr = evaluate_mqar(qm, kv_counts, device=device)
        print(row(bits, qr), flush=True)
        results["quantized"]["full_model"][bits] = qr
        del qm; torch.cuda.empty_cache() if device == "cuda" else None

    # ─── D. Per-parameter sensitivity at 4-bit ───
    print(f"\n{'='*60}")
    print("D. PER-PARAMETER SENSITIVITY (4-bit, one at a time)")
    print(f"{'='*60}")
    print(f"{'Parameter':>18} {'Overall':>9} {'Δ':>7}", flush=True)

    results["quantized"]["per_param_4bit"] = {}
    for pname in ["W_LTM", "V_gs", "V_T0", "beta_tau", "beta_gm",
                   "C_ch", "gamma", "alpha_ppd", "IC_threshold"]:
        qm = quantize_single_param(model, pname, 4)
        qr = evaluate_mqar(qm, kv_counts, device=device)
        d = qr["overall"] - fp["overall"]
        print(f"  {pname:>16} {qr['overall']:>8.1f}% {d:>+6.1f}", flush=True)
        results["quantized"]["per_param_4bit"][pname] = qr
        del qm; torch.cuda.empty_cache() if device == "cuda" else None

    # ─── Summary ───
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  FP: {fp['overall']:.1f}%\n")
    print("  W_LTM only:")
    for b in args.bits:
        r = results["quantized"]["wltm_only"][b]
        print(f"    {b:>2}-bit ({2*b:>2} cells/wt): {r['overall']:.1f}% ({r['overall']-fp['overall']:+.1f})")
    print("\n  Sensitivity (4-bit, sorted):")
    sens = {k: v["overall"] - fp["overall"]
            for k, v in results["quantized"]["per_param_4bit"].items()}
    for k, v in sorted(sens.items(), key=lambda x: x[1]):
        print(f"    {k:>18}: {v:+.1f}%")

    # ─── Save ───
    outpath = args.output or f"quant_d{args.d_model}.json"
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STP-T v3 Post-Training Quantization")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--d_model", type=int, required=True, help="Model dimension")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    run(args)
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
