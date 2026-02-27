#!/usr/bin/env python3
"""
extract_results.py — Aggregate & Analyze STP-T MQAR Phase 1 Results
====================================================================

Three input modes (uses whichever are available, in priority order):
  1. Per-run JSONs from patched train.py  (most complete — has best epoch + history)
  2. WandB offline directories             (good fallback — has full wandb data)
  3. Experiment log file                   (last resort — parsed from tqdm output)

Outputs (all written to --output_dir):
  summary.json            All runs, full detail
  summary.csv             Spreadsheet-friendly, one row per run
  best_per_model.json     Best LR per (model, d_model) — what you cite in the paper
  comparison_table.txt    ASCII table for quick terminal inspection

Usage:
  # Auto-detect (recommended — just point at the results dir)
  python extract_results.py

  # Explicit paths
  python extract_results.py \\
      --runs_dir /workspace/results/runs \\
      --wandb_dir /workspace/zoology/wandb \\
      --log_file /workspace/experiment_log.txt \\
      --output_dir /workspace/results
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ═══════════════════════════════════════════════════════════════
# Source 1: Per-run JSON files (from patched train.py)
# ═══════════════════════════════════════════════════════════════

def extract_from_run_jsons(runs_dir):
    results = []
    runs_path = Path(runs_dir)
    json_files = sorted(runs_path.glob("*.json"))
    print(f"  Found {len(json_files)} run JSON files in {runs_dir}")
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            data["source"] = "run_json"
            data["source_file"] = str(jf)
            results.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse {jf.name}: {e}")
    return results


# ═══════════════════════════════════════════════════════════════
# Source 2: WandB offline directories
# ═══════════════════════════════════════════════════════════════

def extract_from_wandb(wandb_dir):
    results = []
    wandb_path = Path(wandb_dir)
    run_dirs = sorted(wandb_path.glob("offline-run-*"))
    if not run_dirs:
        run_dirs = sorted(wandb_path.glob("run-*"))
    print(f"  Found {len(run_dirs)} wandb run directories in {wandb_dir}")

    for run_dir in run_dirs:
        run_data = {"source": "wandb", "wandb_run_dir": str(run_dir)}

        # Summary
        for loc in [run_dir / "files" / "wandb-summary.json", run_dir / "wandb-summary.json"]:
            if loc.exists():
                try:
                    summary = json.loads(loc.read_text())
                    run_data["final_valid_accuracy"] = summary.get("valid/accuracy")
                    run_data["final_valid_loss"] = summary.get("valid/loss")
                    run_data["train_loss_final"] = summary.get("train/loss")
                    run_data["num_parameters"] = summary.get("num_parameters")
                    run_data["final_epoch"] = summary.get("epoch")
                    acc_by_kv = {}
                    for key, val in summary.items():
                        m = re.match(r"valid/num_kv_pairs/accuracy-(\d+)", key)
                        if m and val is not None:
                            acc_by_kv[int(m.group(1))] = val
                    if acc_by_kv:
                        run_data["final_accuracy_by_kv_pairs"] = acc_by_kv
                except Exception:
                    pass
                break

        # Config
        for loc in [run_dir / "files" / "config.yaml", run_dir / "config.yaml"]:
            if loc.exists():
                try:
                    if HAS_YAML:
                        config = yaml.safe_load(loc.read_text())
                        run_data["run_id"] = config.get("run_id", {}).get("value", "")
                        run_data["sweep_id"] = config.get("sweep_id", {}).get("value", "")
                        run_data["learning_rate"] = config.get("learning_rate", {}).get("value")
                        run_data["max_epochs"] = config.get("max_epochs", {}).get("value")
                        model_cfg = config.get("model", {}).get("value", {})
                        if isinstance(model_cfg, dict):
                            run_data["model_name"] = model_cfg.get("name", "")
                            run_data["d_model"] = model_cfg.get("d_model")
                            run_data["n_layers"] = model_cfg.get("n_layers")
                            run_data["vocab_size"] = model_cfg.get("vocab_size")
                    else:
                        text = loc.read_text()
                        m = re.search(r"run_id:.*?value:\s*(.+)", text)
                        if m: run_data["run_id"] = m.group(1).strip()
                        m = re.search(r"learning_rate:.*?value:\s*([\d.e+-]+)", text)
                        if m: run_data["learning_rate"] = float(m.group(1))
                except Exception:
                    pass
                break

        # History (best-epoch)
        for loc in [run_dir / "files" / "wandb-history.jsonl", run_dir / "wandb-history.jsonl"]:
            if loc.exists():
                try:
                    epochs_data = []
                    with open(loc) as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            if "valid/accuracy" in entry:
                                epochs_data.append(entry)
                    if epochs_data:
                        best = max(epochs_data, key=lambda x: x.get("valid/accuracy", 0))
                        run_data["best_epoch"] = best.get("epoch")
                        run_data["best_valid_accuracy"] = best.get("valid/accuracy")
                        run_data["best_valid_loss"] = best.get("valid/loss")
                        best_kv = {}
                        for key, val in best.items():
                            m = re.match(r"valid/num_kv_pairs/accuracy-(\d+)", key)
                            if m and val is not None:
                                best_kv[int(m.group(1))] = val
                        if best_kv:
                            run_data["best_accuracy_by_kv_pairs"] = best_kv
                except Exception:
                    pass
                break

        if "model_name" not in run_data and "run_id" in run_data:
            _parse_run_id_into(run_data)

        if run_data.get("final_valid_accuracy") is not None or run_data.get("best_valid_accuracy") is not None:
            results.append(run_data)

    return results


# ═══════════════════════════════════════════════════════════════
# Source 3: Experiment log parsing (fallback)
# ═══════════════════════════════════════════════════════════════

def extract_from_log(log_file):
    results = []
    content = Path(log_file).read_text()
    run_ids = re.findall(r"run_id='([^']+)'", content)

    val_pattern = re.compile(
        r"Valid Epoch (\d+)/(\d+).*?"
        r"valid/loss=([\d.]+).*?"
        r"valid/accuracy=([\d.]+)"
        r"(.*?)(?:\n|$)"
    )

    current_run = 0
    prev_epoch = -1
    best_per_run = {}
    last_per_run = {}

    for match in val_pattern.finditer(content):
        epoch = int(match.group(1))
        max_epochs = int(match.group(2))
        if epoch <= prev_epoch:
            current_run += 1
        prev_epoch = epoch

        acc_by_kv = {}
        for kv_match in re.finditer(r"accuracy-(\d+)=([\d.]+)", match.group(5)):
            acc_by_kv[int(kv_match.group(1))] = float(kv_match.group(2))

        entry = {
            "epoch": epoch, "max_epochs": max_epochs,
            "valid_loss": float(match.group(3)),
            "valid_accuracy": float(match.group(4)),
            "accuracy_by_kv_pairs": acc_by_kv,
        }
        last_per_run[current_run] = entry
        if current_run not in best_per_run or entry["valid_accuracy"] > best_per_run[current_run]["valid_accuracy"]:
            best_per_run[current_run] = entry

    print(f"  Found {len(last_per_run)} completed runs in log")

    for i, last in sorted(last_per_run.items()):
        best = best_per_run.get(i, last)
        run_data = {
            "final_valid_accuracy": last["valid_accuracy"],
            "final_valid_loss": last["valid_loss"],
            "final_epoch": last["epoch"],
            "max_epochs": last["max_epochs"],
            "final_accuracy_by_kv_pairs": last["accuracy_by_kv_pairs"],
            "best_valid_accuracy": best["valid_accuracy"],
            "best_valid_loss": best["valid_loss"],
            "best_epoch": best["epoch"],
            "best_accuracy_by_kv_pairs": best["accuracy_by_kv_pairs"],
            "source": "log",
        }
        if i < len(run_ids):
            run_data["run_id"] = run_ids[i]
            _parse_run_id_into(run_data)
        results.append(run_data)

    return results


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _parse_run_id_into(d):
    run_id = d.get("run_id", "")
    parts = run_id.split("-")
    if parts:
        d.setdefault("model_name", parts[0])
        for p in parts[1:]:
            if p.startswith("d") and p[1:].isdigit():
                d.setdefault("d_model", int(p[1:]))
        lr_str = "-".join(parts[1:])
        m = re.search(r"lr([\d.e+-]+)", lr_str)
        if m:
            try:
                d.setdefault("learning_rate", float(m.group(1)))
            except ValueError:
                pass


def merge_results(sources_list):
    priority = {"run_json": 0, "wandb": 1, "log": 2}
    by_run_id = defaultdict(list)
    no_id = []
    for results in sources_list:
        for r in results:
            rid = r.get("run_id")
            if rid:
                by_run_id[rid].append(r)
            else:
                no_id.append(r)
    merged = []
    for rid, entries in by_run_id.items():
        entries.sort(key=lambda x: priority.get(x.get("source", "log"), 99))
        base = entries[0]
        for other in entries[1:]:
            for k, v in other.items():
                if k not in base or base[k] is None:
                    base[k] = v
        merged.append(base)
    merged.extend(no_id)
    return merged


def compute_best_per_model(results):
    grouped = defaultdict(list)
    for r in results:
        key = (r.get("model_name", "?"), r.get("d_model", "?"))
        grouped[key].append(r)

    best = {}
    for (model, d), runs in sorted(grouped.items()):
        best_run = max(runs, key=lambda x: x.get("best_valid_accuracy") or x.get("final_valid_accuracy") or 0)
        acc = best_run.get("best_valid_accuracy") or best_run.get("final_valid_accuracy")
        loss = best_run.get("best_valid_loss") or best_run.get("final_valid_loss")
        kv = best_run.get("best_accuracy_by_kv_pairs") or best_run.get("final_accuracy_by_kv_pairs") or {}

        best[f"{model}_d{d}"] = {
            "model_name": model, "d_model": d,
            "best_lr": best_run.get("learning_rate"),
            "best_valid_accuracy": acc, "best_valid_loss": loss,
            "best_epoch": best_run.get("best_epoch"),
            "accuracy_by_kv_pairs": kv,
            "num_parameters": best_run.get("num_parameters"),
            "run_id": best_run.get("run_id"),
            "checkpoint_path": best_run.get("checkpoint_path"),
            "training_time_seconds": best_run.get("training_time_seconds"),
        }
    return best


def generate_comparison_table(best_per_model):
    lines = []
    lines.append("=" * 135)
    lines.append("STP-T MQAR Phase 1: Best Results Per Model (best LR, best epoch)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 135)

    kv_pairs = [4, 8, 16, 32, 64, 128, 256]
    header = f"{'Model':<22} {'d':>4} {'BestAcc':>7} {'Epoch':>5}  "
    header += "  ".join(f"kv={k:>3}" for k in kv_pairs)
    header += f"  {'LR':>10}  {'Params':>10}  {'Time':>6}"
    lines.append(header)
    lines.append("-" * 135)

    model_order = [
        "attention", "based", "retnet", "gla", "delta_net",
        "gated_delta_net", "mamba2", "stp_v3", "stp_v4"
    ]
    sorted_keys = sorted(
        best_per_model.keys(),
        key=lambda k: (
            model_order.index(best_per_model[k]["model_name"])
            if best_per_model[k]["model_name"] in model_order else 99,
            best_per_model[k].get("d_model", 0)
        )
    )

    prev_model = None
    for key in sorted_keys:
        r = best_per_model[key]
        model = r["model_name"]
        if prev_model and model != prev_model:
            lines.append("")
        prev_model = model

        acc = r.get("best_valid_accuracy", 0) or 0
        epoch = r.get("best_epoch")
        epoch_str = str(epoch) if epoch is not None else "?"
        lr = r.get("best_lr", 0) or 0
        params = r.get("num_parameters")
        params_str = f"{params:,}" if params else "?"
        t = r.get("training_time_seconds")
        time_str = f"{t/60:.0f}m" if t else "?"

        kv_accs = r.get("accuracy_by_kv_pairs", {})
        kv_str = "  ".join(
            f"{kv_accs.get(k, kv_accs.get(str(k), 0))*100:6.1f}%"
            if kv_accs.get(k, kv_accs.get(str(k))) is not None else "     ?"
            for k in kv_pairs
        )

        d = r.get("d_model", "?")
        prefix = ">> " if model.startswith("stp") else "   "
        line = f"{prefix}{model:<19} {d:>4} {acc:>7.4f} {epoch_str:>5}  {kv_str}  {lr:>10.1e}  {params_str:>10}  {time_str:>6}"
        lines.append(line)

    lines.append("-" * 135)
    lines.append(">> = STP models (this work)")
    lines.append("BestAcc = best validation accuracy across all epochs (not just final)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════

def save_all(results, best_per_model, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    p = out / "summary.json"
    with open(p, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  {p} ({len(results)} runs)")

    p = out / "summary.csv"
    if results:
        flat = []
        for r in results:
            row = {}
            for k, v in r.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        row[f"{k}.{k2}"] = v2
                elif isinstance(v, list):
                    continue
                else:
                    row[k] = v
            flat.append(row)
        all_keys = sorted(set().union(*[set(r.keys()) for r in flat]))
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(flat)
        print(f"  {p}")

    p = out / "best_per_model.json"
    with open(p, "w") as f:
        json.dump(best_per_model, f, indent=2, default=str)
    print(f"  {p} ({len(best_per_model)} model x d_model combos)")

    table = generate_comparison_table(best_per_model)
    p = out / "comparison_table.txt"
    with open(p, "w") as f:
        f.write(table)
    print(f"  {p}")
    print()
    print(table)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Aggregate STP-T MQAR Phase 1 results")
    parser.add_argument("--runs_dir", type=str, help="Per-run JSON dir (from patched train.py)")
    parser.add_argument("--wandb_dir", type=str, help="WandB offline runs directory")
    parser.add_argument("--log_file", type=str, help="Experiment log file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if not args.runs_dir and os.path.isdir("/workspace/results/runs"):
        args.runs_dir = "/workspace/results/runs"
    if not args.wandb_dir and os.path.isdir("/workspace/zoology/wandb"):
        args.wandb_dir = "/workspace/zoology/wandb"
    if not args.log_file and os.path.isfile("/workspace/experiment_log.txt"):
        args.log_file = "/workspace/experiment_log.txt"
    if not args.output_dir:
        args.output_dir = "/workspace/results"

    if not any([args.runs_dir, args.wandb_dir, args.log_file]):
        parser.error("No data sources found. Provide --runs_dir, --wandb_dir, or --log_file")

    sources = []
    if args.runs_dir and os.path.isdir(args.runs_dir):
        print(f"[1] Per-run JSONs: {args.runs_dir}")
        sources.append(extract_from_run_jsons(args.runs_dir))
    if args.wandb_dir and os.path.isdir(args.wandb_dir):
        print(f"[2] WandB offline: {args.wandb_dir}")
        sources.append(extract_from_wandb(args.wandb_dir))
    if args.log_file and os.path.isfile(args.log_file):
        print(f"[3] Experiment log: {args.log_file}")
        sources.append(extract_from_log(args.log_file))

    if not sources:
        print("No results found!")
        return

    all_results = merge_results(sources)
    print(f"\nTotal unique runs after merge: {len(all_results)}")
    best = compute_best_per_model(all_results)
    print(f"\nSaving to: {args.output_dir}")
    save_all(all_results, best, args.output_dir)


if __name__ == "__main__":
    main()
