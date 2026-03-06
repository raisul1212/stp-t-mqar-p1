#!/usr/bin/env python3
"""
STP-T v3 Post-Training Quantization Experiment
===============================================

Answers: "How many bits do RRAM cells need for STP-T v3?"

Protocol:
  1. Train v3 at FP16/FP32 (full precision) on MQAR
  2. After training, quantize static components to N bits:
     - W_LTM (dk × dk weight matrix) → RRAM conductances
     - λ (dk × dk decay matrix) → RRAM or SRAM
     - G (dk × dk gate matrix) → RRAM or SRAM
     - sign (dk × dk sign matrix) → RRAM or SRAM
  3. Evaluate quantized model on MQAR (no retraining)
  4. Report accuracy vs bit-width for each component and combined

Quantization modes:
  A. Per-component: quantize one matrix at a time (others stay FP)
  B. All-static: quantize all static matrices simultaneously
  C. All-static + state: also quantize F(t) accumulation (simulates
     limited-precision SRAM for the volatile state)

Bit widths: 2, 3, 4, 5, 6 bits (and FP baseline)

Usage:
  # On RunPod, after setup.sh has been run:
  cd /workspace/zoology
  python3 /workspace/stp-t-mqar-p1/scripts/quantize_v3.py \
      --d_model 128 --bits 2 3 4 5 6 --seeds 3

  # Quick test:
  python3 /workspace/stp-t-mqar-p1/scripts/quantize_v3.py \
      --d_model 128 --bits 4 6 --seeds 1 --quick

Author: RISE Lab, Purdue University
Date: March 2026
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. QUANTIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def uniform_quantize(tensor: torch.Tensor, n_bits: int,
                     symmetric: bool = True) -> torch.Tensor:
    """
    Uniform affine quantization to n_bits.

    For symmetric: maps [−max, +max] → {−2^(n-1), ..., 2^(n-1)−1}
    For asymmetric: maps [min, max] → {0, ..., 2^n − 1}

    Returns: dequantized tensor (float, but snapped to quantization grid)
    """
    if n_bits >= 16:
        return tensor  # No quantization

    n_levels = 2 ** n_bits

    if symmetric:
        abs_max = tensor.abs().max().clamp(min=1e-8)
        scale = abs_max / (n_levels // 2 - 1)
        quantized = torch.round(tensor / scale).clamp(
            -(n_levels // 2), n_levels // 2 - 1
        )
        return quantized * scale
    else:
        t_min = tensor.min()
        t_max = tensor.max()
        scale = (t_max - t_min).clamp(min=1e-8) / (n_levels - 1)
        zero_point = torch.round(-t_min / scale).clamp(0, n_levels - 1)
        quantized = torch.round(tensor / scale + zero_point).clamp(0, n_levels - 1)
        return (quantized - zero_point) * scale


def quantize_bounded(tensor: torch.Tensor, n_bits: int,
                     low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    """
    Quantize a tensor known to be in [low, high] to n_bits uniform levels.

    This is appropriate for λ ∈ (0,1), G ∈ (0, ∞) after clamping, etc.
    Uses asymmetric quantization within the known range.
    """
    if n_bits >= 16:
        return tensor

    n_levels = 2 ** n_bits
    scale = (high - low) / (n_levels - 1)
    quantized = torch.round((tensor - low) / scale).clamp(0, n_levels - 1)
    return quantized * scale + low


def quantize_state_accumulation(F_prev: torch.Tensor, update: torch.Tensor,
                                retention: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Simulate quantized state accumulation:
      F(t) = quantize(retention * F(t-1) + update)

    This models limited-precision SRAM for the volatile state F(t).
    """
    F_new = retention * F_prev + update
    if n_bits >= 16:
        return F_new
    return uniform_quantize(F_new, n_bits, symmetric=True)


# ═══════════════════════════════════════════════════════════════
# 2. STP-T v3 PHYSICS (standalone, no Zoology dependency)
# ═══════════════════════════════════════════════════════════════

def compute_stp_physics(W_LTM, V_gs, V_T0, U_T, beta_tau, beta_gm,
                        C_ch, alpha, IC_threshold):
    """
    Compute static physics quantities from v3 parameters.
    Returns: lambda_, G, sign_mat (all H × dk × dk)
    """
    # V_eff: (H, dk, dk) = scalar + scalar + (H, dk, dk)
    V_eff = V_gs.view(-1, 1, 1) - V_T0.view(-1, 1, 1) + W_LTM

    # Channel conductance
    x = V_eff / U_T
    g_ch = beta_tau.view(-1, 1, 1) * F.softplus(x)

    # Decay: λ = 1 - exp(-g_ch * dt / C_ch),  dt=1
    lambda_ = 1.0 - torch.exp(-g_ch / C_ch.view(-1, 1, 1))

    # Transconductance gate: G = beta_gm * softplus * sigmoid
    G = beta_gm.view(-1, 1, 1) * F.softplus(x) * torch.sigmoid(x)

    # Sign: tanh(alpha * (g_ch - threshold))
    sign_mat = torch.tanh(alpha * (g_ch - IC_threshold))

    return lambda_, G, sign_mat


class STPv3Standalone(nn.Module):
    """
    Standalone STP-T v3 for quantization experiments.
    Matches the Zoology mixer exactly but without Zoology dependencies.
    """

    def __init__(self, d_model: int, num_heads: int = 2, vocab_size: int = 8192,
                 n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(STPv3Layer(d_model, num_heads, self.dk))

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor,
                state_quant_bits: int = 16) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
            state_quant_bits: bit-width for F(t) accumulation (16 = FP)
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embed(input_ids)  # (B, T, d_model)
        for layer in self.layers:
            x = layer(x, state_quant_bits=state_quant_bits)
        x = self.ln_f(x)
        return self.head(x)


class STPv3Layer(nn.Module):
    """Single STP-T v3 transformer block (BaseConv + STP mixer)."""

    def __init__(self, d_model: int, num_heads: int, dk: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = dk
        H = num_heads

        # Short convolution (matches Zoology BaseConv)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2,
                              groups=d_model)

        # QKV projection
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # STP physics parameters
        self.W_LTM = nn.Parameter(torch.randn(H, dk, dk) * 0.02)
        self.V_T0 = nn.Parameter(torch.zeros(H))
        self.V_gs = nn.Parameter(torch.ones(H) * 0.5)
        self.beta_tau = nn.Parameter(torch.ones(H))
        self.beta_gm = nn.Parameter(torch.ones(H) * 0.1)
        self.C_ch = nn.Parameter(torch.ones(H))
        self.gamma = nn.Parameter(torch.ones(H) * 0.1)
        self.alpha_ppd = nn.Parameter(torch.tensor(1.0))
        self.IC_threshold = nn.Parameter(torch.tensor(0.5))
        self.U_T = 1.0  # fixed thermal voltage

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(dk)

        # Pre-computed physics (set during forward or quantization)
        self._lambda = None
        self._G = None
        self._sign = None

    def precompute_physics(self):
        """Compute and cache static physics quantities."""
        self._lambda, self._G, self._sign = compute_stp_physics(
            self.W_LTM, self.V_gs, self.V_T0, self.U_T,
            self.beta_tau, self.beta_gm, self.C_ch,
            self.alpha_ppd, self.IC_threshold
        )

    def forward(self, x: torch.Tensor, state_quant_bits: int = 16) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.num_heads, self.dk

        residual = x
        x = self.ln1(x)

        # Short convolution (causal)
        x_conv = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x = x + x_conv

        # QKV
        qkv = self.W_qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to heads
        q = q.view(B, T, H, dk).transpose(1, 2)  # (B, H, T, dk)
        k = k.view(B, T, H, dk).transpose(1, 2)
        v = v.view(B, T, H, dk).transpose(1, 2)

        # Key normalization
        k = self.ln_k(k)

        # Compute physics (or use cached)
        if self._lambda is None:
            self.precompute_physics()

        lambda_ = self._lambda  # (H, dk, dk)
        G = self._G
        sign_mat = self._sign
        gamma = self.gamma.view(1, H, 1, 1)

        # Recurrence
        retention = (1.0 - lambda_).unsqueeze(0)  # (1, H, dk, dk)
        gate = (gamma * sign_mat.unsqueeze(0) *
                G.unsqueeze(0))  # (1, H, dk, dk)

        F_state = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            q_t = q[:, :, t, :]  # (B, H, dk)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            # Outer product
            outer = torch.einsum('bhv,bhk->bhvk', v_t, k_t)  # (B, H, dk, dk)

            # State update
            update = gate * outer
            F_new = retention * F_state + update

            # Quantize state if requested
            if state_quant_bits < 16:
                F_new = uniform_quantize(F_new, state_quant_bits, symmetric=True)

            F_state = F_new

            # Output: y = (W_LTM + F) · q
            W_eff = self.W_LTM.unsqueeze(0) + F_state  # (B, H, dk, dk)
            y_t = torch.einsum('bhvk,bhk->bhv', W_eff, q_t)  # (B, H, dk)
            outputs.append(y_t)

        # Stack and reshape
        y = torch.stack(outputs, dim=2)  # (B, H, T, dk)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        y = self.W_o(y)

        return residual + y


# ═══════════════════════════════════════════════════════════════
# 3. QUANTIZATION SWEEP
# ═══════════════════════════════════════════════════════════════

def quantize_model_static(model: STPv3Standalone, bits: int,
                          components: str = "all") -> STPv3Standalone:
    """
    Post-training quantization of v3 static components.

    Args:
        model: trained FP model
        bits: target bit-width (2-6)
        components: which to quantize
            "wltm"   - only W_LTM
            "lambda"  - only λ (after physics computation)
            "gate"   - only G
            "sign"   - only sign
            "all"    - all static matrices
    Returns:
        quantized model (deep copy, original unchanged)
    """
    # Clear cached non-leaf tensors before deepcopy (they break copy.deepcopy)
    for layer in model.layers:
        layer._lambda = None
        layer._G = None
        layer._sign = None

    qmodel = copy.deepcopy(model)

    for layer in qmodel.layers:
        # First compute physics at full precision
        layer.precompute_physics()

        with torch.no_grad():
            # Quantize W_LTM
            if components in ("wltm", "all"):
                layer.W_LTM.data = uniform_quantize(
                    layer.W_LTM.data, bits, symmetric=True
                )

            # Quantize pre-computed physics matrices
            if components in ("lambda", "all"):
                # λ ∈ (0, 1) → use bounded quantization
                layer._lambda = quantize_bounded(
                    layer._lambda, bits, low=0.0, high=1.0
                )

            if components in ("gate", "all"):
                # G ≥ 0 → use asymmetric
                g_max = layer._G.max().item()
                layer._G = quantize_bounded(
                    layer._G, bits, low=0.0, high=g_max
                )

            if components in ("sign", "all"):
                # sign ∈ (-1, 1) → use symmetric
                layer._sign = uniform_quantize(
                    layer._sign, bits, symmetric=True
                )

    return qmodel


# ═══════════════════════════════════════════════════════════════
# 4. MQAR DATA GENERATION (standalone, matches Zoology)
# ═══════════════════════════════════════════════════════════════

def generate_mqar_batch(batch_size: int, n_kv_pairs: int,
                        vocab_size: int = 8192, device: str = "cuda"):
    """
    Generate MQAR batch matching Zoology's multiquery_ar format.

    Sequence: [k1, v1, k2, v2, ..., kN, vN, PAD..., q1, q2, ..., qN]
    Target:   [0,  0,  0,  0,  ..., 0,  0,  PAD..., v1, v2, ..., vN]

    Returns dict with input_ids, targets, mask (query positions only)
    """
    # Sequence length: 2*N for KV pairs + N for queries + some padding
    seq_len = max(64, 4 * n_kv_pairs)  # matches Zoology scaling

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for b in range(batch_size):
        # Generate unique keys
        keys = torch.randperm(vocab_size, device=device)[:n_kv_pairs]
        values = torch.randint(0, vocab_size, (n_kv_pairs,), device=device)

        # Fill KV pairs
        for i in range(n_kv_pairs):
            input_ids[b, 2 * i] = keys[i]
            input_ids[b, 2 * i + 1] = values[i]

        # Fill queries (shuffled order)
        query_start = seq_len - n_kv_pairs
        perm = torch.randperm(n_kv_pairs, device=device)
        for i in range(n_kv_pairs):
            qi = perm[i].item()
            input_ids[b, query_start + i] = keys[qi]
            targets[b, query_start + i] = values[qi]
            mask[b, query_start + i] = True

    return {"input_ids": input_ids, "targets": targets, "mask": mask}


# ═══════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_v3(d_model: int = 128, num_heads: int = 2, n_layers: int = 2,
             vocab_size: int = 8192, lr: float = 3.2e-3,
             n_epochs: int = 32, batch_size: int = 256,
             device: str = "cuda", seed: int = 42) -> STPv3Standalone:
    """Train STP-T v3 on MQAR (matching Phase 1 setup)."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = STPv3Standalone(
        d_model=d_model, num_heads=num_heads,
        vocab_size=vocab_size, n_layers=n_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # Cosine schedule
    total_steps = n_epochs * 400  # ~100K examples / batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # Training KV counts — reduced dataset for quantization experiments.
    # Full Zoology uses 100K/20K per KV; we use 10K/5K which trains in
    # ~2 min/epoch on A100 instead of ~33 min/epoch.
    # The model still reaches >80% accuracy, sufficient for measuring
    # quantization degradation (we care about relative drop, not absolute peak).
    train_kv_counts = [4, 8, 16, 32, 64]
    train_samples_per_kv = {4: 10000, 8: 5000, 16: 5000, 32: 5000, 64: 5000}

    print(f"\n{'='*60}")
    print(f"Training STP-T v3: d={d_model}, H={num_heads}, seed={seed}")
    print(f"{'='*60}")

    model.train()
    step = 0
    best_loss = float('inf')
    nan_detected = False

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_steps = 0

        for kv in train_kv_counts:
            n_batches = max(1, train_samples_per_kv[kv] // batch_size)
            for batch_idx in range(n_batches):
                batch = generate_mqar_batch(batch_size, kv, vocab_size, device)
                logits = model(batch["input_ids"])

                # Loss only on query positions
                loss = F.cross_entropy(
                    logits[batch["mask"]],
                    batch["targets"][batch["mask"]]
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # Zero out NaN gradients
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.zero_()

                optimizer.step()
                scheduler.step()

                # Clear cached physics (parameters may have changed)
                for layer in model.layers:
                    layer._lambda = None
                    layer._G = None
                    layer._sign = None

                loss_val = loss.item()
                if not np.isfinite(loss_val):
                    print(f"\n  NaN detected at epoch {epoch}, kv={kv}. Stopping.")
                    nan_detected = True
                    break

                epoch_loss += loss_val
                epoch_steps += 1
                step += 1

            if nan_detected:
                break
        if nan_detected:
            break

        avg_loss = epoch_loss / max(epoch_steps, 1)
        # Print every epoch for visibility
        print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  steps={epoch_steps}")

    if nan_detected:
        print(f"  Training stopped early due to NaN. Best loss: {best_loss:.4f}")
    else:
        print(f"  Training complete. Final loss: {avg_loss:.4f}")
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_mqar(model: STPv3Standalone, kv_counts: List[int],
                  vocab_size: int = 8192, n_eval: int = 1000,
                  batch_size: int = 128, device: str = "cuda",
                  state_quant_bits: int = 16) -> Dict:
    """
    Evaluate on MQAR at multiple KV pair counts.

    Returns:
        {
            "overall": float,
            "per_kv": {4: float, 8: float, ...},
        }
    """
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

            # Clear cached physics to force recompute
            for layer in model.layers:
                if layer._lambda is None:
                    layer.precompute_physics()

            logits = model(batch["input_ids"],
                           state_quant_bits=state_quant_bits)
            preds = logits[batch["mask"]].argmax(dim=-1)
            targets = batch["targets"][batch["mask"]]
            correct += (preds == targets).sum().item()
            count += targets.numel()

        acc = 100.0 * correct / max(count, 1)
        results[kv] = acc
        total_correct += correct
        total_count += count

    overall = 100.0 * total_correct / max(total_count, 1)
    return {"overall": overall, "per_kv": results}


# ═══════════════════════════════════════════════════════════════
# 7. MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════

def run_quantization_sweep(args):
    """Run the full quantization experiment.
    
    Strategy: Train ONCE per seed, save checkpoint, then run all
    quantization evaluations from the saved model. Evaluation is fast
    (seconds per bit-width); only training is slow.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    kv_counts = [4, 8, 16, 32, 64, 128, 256]
    n_epochs = 8 if args.quick else 32
    bs = 64 if args.d_model >= 256 else 256

    all_results = {
        "config": {
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "n_layers": args.n_layers,
            "bits_tested": args.bits,
            "seeds": args.seeds,
            "n_epochs": n_epochs,
            "vocab_size": 8192,
        },
        "runs": []
    }

    # ─── Phase 1: Train models (slow) ───────────────────────────
    # Train one model per seed, save checkpoints.
    # If checkpoint already exists, skip training and load it.

    ckpt_dir = Path(args.output).parent if args.output else Path(".")
    models = {}

    for seed in range(args.seeds):
        ckpt_path = ckpt_dir / f"quant_model_d{args.d_model}_seed{seed}.pt"

        if ckpt_path.exists() and not args.retrain:
            print(f"\n{'#'*60}")
            print(f"# SEED {seed} — Loading checkpoint: {ckpt_path}")
            print(f"{'#'*60}")
            model = STPv3Standalone(
                d_model=args.d_model, num_heads=args.num_heads,
                vocab_size=8192, n_layers=args.n_layers
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            print(f"  Loaded. Skipping training.")
        else:
            print(f"\n{'#'*60}")
            print(f"# SEED {seed} — Training from scratch")
            print(f"{'#'*60}")
            model = train_v3(
                d_model=args.d_model,
                num_heads=args.num_heads,
                n_layers=args.n_layers,
                lr=3.2e-3,
                n_epochs=n_epochs,
                batch_size=bs,
                device=device,
                seed=seed,
            )
            # Save checkpoint
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        models[seed] = model

    # ─── Phase 2: Quantization evaluations (fast) ──────────────
    # All bit-widths × all conditions × all seeds.
    # This is pure evaluation — no training, takes minutes total.

    print(f"\n{'='*60}")
    print(f"QUANTIZATION EVALUATION ({len(args.bits)} bit-widths × 4 conditions × {args.seeds} seeds)")
    print(f"{'='*60}")

    for seed in range(args.seeds):
        model = models[seed]

        print(f"\n{'#'*60}")
        print(f"# SEED {seed} — Evaluating")
        print(f"{'#'*60}")

        # ─── FP Baseline ───
        print(f"\nFP baseline...")
        fp_results = evaluate_mqar(model, kv_counts, device=device)
        print(f"  FP overall: {fp_results['overall']:.1f}%")
        for kv, acc in fp_results["per_kv"].items():
            print(f"    kv={kv}: {acc:.1f}%")

        seed_results = {
            "seed": seed,
            "fp_baseline": fp_results,
            "quantized": {}
        }

        # A. All-static quantization (the main RRAM experiment)
        print(f"\n{'─'*60}")
        print(f"A. ALL-STATIC QUANTIZATION (W_LTM + λ + G + sign)")
        print(f"{'─'*60}")
        print(f"{'Bits':>6} {'Overall':>10} {'Δ':>8}  "
              f"{'kv4':>6} {'kv8':>6} {'kv16':>6} {'kv32':>6} "
              f"{'kv64':>6} {'kv128':>6} {'kv256':>6}")

        seed_results["quantized"]["all_static"] = {}
        for bits in args.bits:
            # Clear cached physics before deepcopy
            for layer in model.layers:
                layer._lambda = None; layer._G = None; layer._sign = None

            qmodel = quantize_model_static(model, bits, components="all")
            qmodel = qmodel.to(device)
            qresults = evaluate_mqar(qmodel, kv_counts, device=device)

            delta = qresults["overall"] - fp_results["overall"]
            kv_str = "  ".join(
                f"{qresults['per_kv'].get(kv, 0):5.1f}" for kv in kv_counts
            )
            print(f"  {bits:>4}b {qresults['overall']:>9.1f}% {delta:>+7.1f}  {kv_str}")

            seed_results["quantized"]["all_static"][bits] = qresults
            del qmodel
            if device == "cuda": torch.cuda.empty_cache()

        # B. Per-component quantization
        print(f"\n{'─'*60}")
        print(f"B. PER-COMPONENT QUANTIZATION (isolate sensitivity)")
        print(f"{'─'*60}")

        for component in ["wltm", "lambda", "gate", "sign"]:
            print(f"\n  Component: {component}")
            print(f"  {'Bits':>6} {'Overall':>10} {'Δ':>8}")
            seed_results["quantized"][component] = {}

            for bits in args.bits:
                for layer in model.layers:
                    layer._lambda = None; layer._G = None; layer._sign = None

                qmodel = quantize_model_static(model, bits, components=component)
                qmodel = qmodel.to(device)
                qresults = evaluate_mqar(qmodel, kv_counts, device=device)
                delta = qresults["overall"] - fp_results["overall"]
                print(f"    {bits:>4}b {qresults['overall']:>9.1f}% {delta:>+7.1f}")
                seed_results["quantized"][component][bits] = qresults
                del qmodel
                if device == "cuda": torch.cuda.empty_cache()

        # C. State quantization (F(t) in limited-precision SRAM)
        print(f"\n{'─'*60}")
        print(f"C. STATE QUANTIZATION (F(t) accumulation precision)")
        print(f"{'─'*60}")
        print(f"{'Bits':>6} {'Overall':>10} {'Δ':>8}")

        seed_results["quantized"]["state"] = {}
        for bits in args.bits:
            for layer in model.layers:
                layer._lambda = None; layer._G = None; layer._sign = None

            fp_model_copy = copy.deepcopy(model).to(device)
            for layer in fp_model_copy.layers:
                layer.precompute_physics()

            qresults = evaluate_mqar(
                fp_model_copy, kv_counts, device=device,
                state_quant_bits=bits
            )
            delta = qresults["overall"] - fp_results["overall"]
            print(f"  {bits:>4}b {qresults['overall']:>9.1f}% {delta:>+7.1f}")
            seed_results["quantized"]["state"][bits] = qresults
            del fp_model_copy

        # D. Combined: all-static + state quantization
        print(f"\n{'─'*60}")
        print(f"D. FULL SYSTEM (all static @ N bits + state @ N bits)")
        print(f"{'─'*60}")
        print(f"{'Bits':>6} {'Overall':>10} {'Δ':>8}")

        seed_results["quantized"]["full_system"] = {}
        for bits in args.bits:
            for layer in model.layers:
                layer._lambda = None; layer._G = None; layer._sign = None

            qmodel = quantize_model_static(model, bits, components="all")
            qmodel = qmodel.to(device)
            qresults = evaluate_mqar(
                qmodel, kv_counts, device=device,
                state_quant_bits=bits
            )
            delta = qresults["overall"] - fp_results["overall"]
            print(f"  {bits:>4}b {qresults['overall']:>9.1f}% {delta:>+7.1f}")
            seed_results["quantized"]["full_system"][bits] = qresults
            del qmodel

        all_results["runs"].append(seed_results)

        # Save intermediate results after each seed
        outpath = args.output or f"quant_results_d{args.d_model}.json"
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Intermediate results saved: {outpath}")

    # Clean up models
    del models
    if device == "cuda": torch.cuda.empty_cache()

    # ─── Aggregate across seeds ───
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({args.seeds} seeds)")
    print(f"{'='*60}")

    agg = aggregate_results(all_results)
    all_results["aggregate"] = agg

    # Print summary table
    print(f"\n  ALL-STATIC QUANTIZATION (mean ± std across seeds):")
    print(f"  {'Bits':>6} {'Overall':>15} {'kv64':>12} {'kv128':>12} {'kv256':>12}")
    for bits_str, data in agg["all_static"].items():
        bits = int(bits_str)
        overall = f"{data['overall_mean']:.1f}±{data['overall_std']:.1f}"
        kv64 = f"{data.get('kv_64_mean', 0):.1f}±{data.get('kv_64_std', 0):.1f}"
        kv128 = f"{data.get('kv_128_mean', 0):.1f}±{data.get('kv_128_std', 0):.1f}"
        kv256 = f"{data.get('kv_256_mean', 0):.1f}±{data.get('kv_256_std', 0):.1f}"
        print(f"    {bits:>4}b {overall:>15} {kv64:>12} {kv128:>12} {kv256:>12}")

    fp_overall = np.mean([r["fp_baseline"]["overall"] for r in all_results["runs"]])
    print(f"      FP {'':>3} {fp_overall:>9.1f}%")

    # ─── Save final ───
    outpath = args.output or f"quant_results_d{args.d_model}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved to: {outpath}")

    return all_results


def aggregate_results(all_results: dict) -> dict:
    """Compute mean/std across seeds for each quantization config."""
    agg = {}
    runs = all_results["runs"]

    for qtype in ["all_static", "wltm", "lambda", "gate", "sign", "state", "full_system"]:
        agg[qtype] = {}
        for run in runs:
            if qtype not in run["quantized"]:
                continue
            for bits_str, results in run["quantized"][qtype].items():
                bits_str = str(bits_str)
                if bits_str not in agg[qtype]:
                    agg[qtype][bits_str] = {"overall": [], "per_kv": {}}
                agg[qtype][bits_str]["overall"].append(results["overall"])
                for kv, acc in results["per_kv"].items():
                    kv_key = str(kv)
                    if kv_key not in agg[qtype][bits_str]["per_kv"]:
                        agg[qtype][bits_str]["per_kv"][kv_key] = []
                    agg[qtype][bits_str]["per_kv"][kv_key].append(acc)

        # Compute stats
        for bits_str in list(agg[qtype].keys()):
            data = agg[qtype][bits_str]
            data["overall_mean"] = float(np.mean(data["overall"]))
            data["overall_std"] = float(np.std(data["overall"]))
            for kv_key, accs in data["per_kv"].items():
                data[f"kv_{kv_key}_mean"] = float(np.mean(accs))
                data[f"kv_{kv_key}_std"] = float(np.std(accs))

    return agg


# ═══════════════════════════════════════════════════════════════
# 8. CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STP-T v3 Post-Training Quantization for RRAM CIM"
    )
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension (default: 128)")
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                        help="Bit widths to test")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs, fewer eval samples")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if checkpoint exists")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    t0 = time.time()
    results = run_quantization_sweep(args)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
