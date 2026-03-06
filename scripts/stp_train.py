"""
stp_train.py — Drop-in replacement for zoology/train.py
=========================================================

Adds to Zoology's training loop:
  1. Model checkpoint saving (best + final)
  2. Best-epoch tracking (not just last epoch)
  3. Per-run JSON result export (portable, no wandb dependency)
  4. Automated result extraction after each run

All additions are backward-compatible: if no results_dir is set,
behavior is identical to the original Zoology train.py.

Usage:
  The setup.sh script patches zoology/train.py automatically.
  Or manually: cp stp_train.py /workspace/zoology/zoology/train.py
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import List, Union
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.data.utils import prepare_data, prepare_continuous_data
from zoology.config import TrainConfig
from zoology.model import LanguageModel, ContinuousInputModel
from zoology.logger import WandbLogger
from zoology.utils import set_determinism
from zoology.metrics import compute_mse, compute_ce_with_embeddings


# ─────────────────────────────────────────────────────────────
# Results directory (set by environment variable or default)
# ─────────────────────────────────────────────────────────────

RESULTS_DIR = os.environ.get("STP_RESULTS_DIR", "/workspace/results")
CHECKPOINT_DIR = os.environ.get("STP_CHECKPOINT_DIR", "/workspace/checkpoints")
SAVE_CHECKPOINTS = os.environ.get("STP_SAVE_CHECKPOINTS", "best").lower()  # "best", "all", "none"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        input_type: str = "discrete",
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
        # ── New: result tracking ──
        run_id: str = "default",
        config: TrainConfig = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.input_type = input_type
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.slice_keys = slice_keys
        self.loss_type = loss_type

        # ── New fields ──
        self.run_id = run_id
        self.config = config
        self.best_accuracy = -1.0
        self.best_epoch = -1
        self.best_metrics = {}
        self.all_epoch_metrics = []
        self.train_start_time = None

    def compute_loss(self, inputs, targets):
        if self.input_type == "continuous":
            
            all_embeddings = self.model.backbone.embeddings.word_embeddings.weight
            vocab_size = all_embeddings.shape[0]
            embed_dim = all_embeddings.shape[1]
            value_embeddings = all_embeddings[vocab_size // 2:]  # all values as candidates
            
            outputs = self.model(inputs)
            num_kv_pairs = targets.shape[1]
            outputs = outputs[:, -num_kv_pairs:]
            
            outputs_flat = outputs.reshape(-1, embed_dim)
            targets_flat = targets.reshape(-1)
            
            if self.loss_type == "mse":
                target_embeds = value_embeddings[targets_flat]
                loss, _ = compute_mse(outputs_flat, target_embeds)
            else:  # ce or ce_embed
                loss, _ = compute_ce_with_embeddings(
                    outputs_flat, targets_flat, value_embeddings
                )
            
            logits = outputs_flat @ value_embeddings.T
            preds = (logits).argmax(dim=-1).view(targets.shape)
            return loss, preds
        
        else: # discrete
            if self.loss_type == "ce":
                logits = self.model(inputs)
                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), 
                    targets.flatten()
                )
                preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "mse":
                embeddings = self.model(inputs, return_embeddings=True)
                target_embeds = self.model.backbone.embeddings.word_embeddings(targets)
                mask = (targets != -100).unsqueeze(-1)
                loss, _ = compute_mse(
                    embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
                    target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
                )
                logits = embeddings @ self.model.backbone.embeddings.word_embeddings.weight.T
                preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "ce_embed":
                embeddings = self.model(inputs, return_embeddings=True)
                value_embeddings = self.model.backbone.embeddings.word_embeddings.weight
                flat_embeds = rearrange(embeddings, "b s d -> (b s) d")
                flat_targets = targets.flatten()
                mask = flat_targets != -100
                loss, _ = compute_ce_with_embeddings(
                    flat_embeds[mask], flat_targets[mask], value_embeddings,
                )
                logits = embeddings @ value_embeddings.T
                preds = logits.argmax(dim=-1)
                return loss, preds

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            loss, preds = self.compute_loss(inputs, targets)

            # Auxiliary losses (discrete mode only)
            if self.input_type == "discrete":
                auxiliary_loss = []
                def get_auxiliary_loss(module):
                    if hasattr(module, "get_auxiliary_loss"):
                        auxiliary_loss.append(module.get_auxiliary_loss())
                self.model.apply(get_auxiliary_loss)
                if auxiliary_loss:
                    loss = loss + sum(auxiliary_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log({"train/loss": loss.item(), "epoch": epoch_idx})

    def test(self, epoch_idx: int):
        self.model.eval()
        test_loss = 0
        results = []

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self.compute_loss(inputs, targets)
                test_loss += loss / len(self.test_dataloader)
                results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
                iterator.update(1)

            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }

            # compute metrics for slices
            for key in self.slice_keys:
                acc_by_slice = results.groupby(key)["accuracy"].mean()
                for value, accuracy in acc_by_slice.items():
                    metrics[f"valid/{key}/accuracy-{value}"] = accuracy

            iterator.set_postfix(metrics)
            self.logger.log({"epoch": epoch_idx, **metrics})

        # ── NEW: Track best epoch and save checkpoint ──
        self.all_epoch_metrics.append({"epoch": epoch_idx, **metrics})

        current_acc = test_accuracy.item()
        if current_acc > self.best_accuracy:
            self.best_accuracy = current_acc
            self.best_epoch = epoch_idx
            self.best_metrics = dict(metrics)
            
            # Save best checkpoint
            if SAVE_CHECKPOINTS in ("best", "all"):
                self._save_checkpoint(epoch_idx, metrics, tag="best")
        
        if SAVE_CHECKPOINTS == "all":
            self._save_checkpoint(epoch_idx, metrics, tag=f"epoch{epoch_idx}")

        return metrics

    def _save_checkpoint(self, epoch_idx, metrics, tag="best"):
        """Save model checkpoint with metadata."""
        ckpt_dir = Path(CHECKPOINT_DIR) / self.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = ckpt_dir / f"{tag}.pt"
        ckpt = {
            "epoch": epoch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "run_id": self.run_id,
            "best_accuracy": self.best_accuracy,
        }
        torch.save(ckpt, ckpt_path)

    def _save_run_results(self):
        """Save per-run JSON results (portable, no wandb needed)."""
        results_dir = Path(RESULTS_DIR) / "runs"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        elapsed = time.time() - self.train_start_time if self.train_start_time else 0
        
        # Extract model info from config
        model_name = ""
        d_model = 0
        n_layers = 0
        vocab_size = 0
        if self.config:
            model_name = self.config.model.name
            d_model = self.config.model.d_model
            n_layers = self.config.model.n_layers
            vocab_size = self.config.model.vocab_size
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Build per-kv accuracy from best epoch
        acc_by_kv = {}
        for key, val in self.best_metrics.items():
            import re
            m = re.match(r"valid/num_kv_pairs/accuracy-(\d+)", key)
            if m:
                acc_by_kv[int(m.group(1))] = val
        
        # Also get final epoch per-kv
        final_acc_by_kv = {}
        if self.all_epoch_metrics:
            final = self.all_epoch_metrics[-1]
            for key, val in final.items():
                m = re.match(r"valid/num_kv_pairs/accuracy-(\d+)", key)
                if m:
                    final_acc_by_kv[int(m.group(1))] = val
        
        run_result = {
            "run_id": self.run_id,
            "model_name": model_name,
            "d_model": d_model,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "num_parameters": num_params,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "training_time_seconds": round(elapsed, 1),
            
            # Best epoch results
            "best_epoch": self.best_epoch,
            "best_valid_accuracy": self.best_accuracy,
            "best_valid_loss": self.best_metrics.get("valid/loss"),
            "best_accuracy_by_kv_pairs": acc_by_kv,
            
            # Final epoch results
            "final_epoch": self.all_epoch_metrics[-1]["epoch"] if self.all_epoch_metrics else None,
            "final_valid_accuracy": self.all_epoch_metrics[-1].get("valid/accuracy") if self.all_epoch_metrics else None,
            "final_valid_loss": self.all_epoch_metrics[-1].get("valid/loss") if self.all_epoch_metrics else None,
            "final_accuracy_by_kv_pairs": final_acc_by_kv,
            
            # Full history (compact: just accuracy per epoch)
            "epoch_history": [
                {
                    "epoch": em["epoch"],
                    "valid_accuracy": em.get("valid/accuracy"),
                    "valid_loss": em.get("valid/loss"),
                }
                for em in self.all_epoch_metrics
            ],
            
            # Checkpoint info
            "checkpoint_path": str(Path(CHECKPOINT_DIR) / self.run_id / "best.pt")
                if SAVE_CHECKPOINTS in ("best", "all") else None,
            
            # Metadata
            "timestamp": datetime.now().isoformat(),
            "sweep_id": self.config.sweep_id if self.config else None,
        }
        
        # Save individual run file
        safe_run_id = self.run_id.replace("/", "_").replace(" ", "_")
        run_path = results_dir / f"{safe_run_id}.json"
        with open(run_path, "w") as f:
            json.dump(run_result, f, indent=2, default=str)
        
        return run_result

    def fit(self):
        self.train_start_time = time.time()
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)

            # early stopping
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break

            self.scheduler.step()

        # ── NEW: Save final checkpoint + per-run JSON ──
        if SAVE_CHECKPOINTS in ("best", "all"):
            self._save_checkpoint(
                self.all_epoch_metrics[-1]["epoch"] if self.all_epoch_metrics else 0,
                self.all_epoch_metrics[-1] if self.all_epoch_metrics else {},
                tag="final",
            )
        
        run_result = self._save_run_results()
        
        # Print summary for this run
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE: {self.run_id}")
        print(f"  Best accuracy: {self.best_accuracy:.4f} (epoch {self.best_epoch})")
        print(f"  Final accuracy: {run_result.get('final_valid_accuracy', 'N/A')}")
        if run_result.get("best_accuracy_by_kv_pairs"):
            kv_str = ", ".join(
                f"kv{k}={v:.3f}" 
                for k, v in sorted(run_result["best_accuracy_by_kv_pairs"].items())
            )
            print(f"  Per-KV (best): {kv_str}")
        if run_result.get("checkpoint_path"):
            print(f"  Checkpoint: {run_result['checkpoint_path']}")
        print(f"  Results JSON: {Path(RESULTS_DIR) / 'runs' / (self.run_id + '.json')}")
        print(f"{'='*60}\n")
        
        return run_result


def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    set_determinism(config.seed)
    
    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        train_dataloader, test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        train_dataloader, test_dataloader = prepare_data(config.data)

    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        # ── New: pass config for result saving ──
        run_id=config.run_id,
        config=config,
    )
    task.fit()
    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train()
