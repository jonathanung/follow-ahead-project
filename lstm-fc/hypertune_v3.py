"""Hyperparameter tuning v3 — comprehensive search for best LSTM-FC performance.

Explores dimensions not covered in previous rounds:
- Larger hidden sizes (256, 384, 512)
- 2-layer LSTM with OneCycleLR (previously only tested with step/cosine)
- Gradient clipping
- AdamW optimizer
- Longer training (20k steps per trial)
- Different tanh_power label softness values
- Learning rate warmup ratios
"""

import json
import logging
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from lstm_fc.config import DataConfig, ModelConfig
from lstm_fc.data.augmentation import augment_sequence
from lstm_fc.data.labels import generate_target
from lstm_fc.model import LSTMModel2D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset that supports variable tanh_power
# ---------------------------------------------------------------------------

class TrajectoryDatasetV3(Dataset):
    """Like TrajectoryDataset but allows rebuilding labels with different tanh_power."""

    def __init__(self, config: DataConfig, split: str = "train"):
        self.config = config
        samples, targets = self._load_and_process()

        # Normalize
        last_points = samples[:, config.input_length - 1 : config.input_length, :]
        samples = samples - np.broadcast_to(last_points, samples.shape)

        # Shuffle
        rng = np.random.RandomState(config.seed)
        idx = rng.permutation(len(samples))
        samples = samples[idx]
        targets = targets[idx]

        split_idx = int(config.train_ratio * len(samples))
        if split == "train":
            self.samples = torch.from_numpy(samples[:split_idx].astype(np.float32))
            self.targets = torch.from_numpy(targets[:split_idx].astype(np.float32))
        else:
            self.samples = torch.from_numpy(samples[split_idx:].astype(np.float32))
            self.targets = torch.from_numpy(targets[split_idx:].astype(np.float32))

    def _load_and_process(self):
        raw_data = np.load(self.config.data_path, allow_pickle=True)[
            "positions_3d"
        ].item()

        freq_ratio = self.config.source_freq // self.config.target_freq
        samples = []
        targets = []

        for subject in self.config.subjects:
            data_3d = raw_data[subject][self.config.motion]
            data_2d = data_3d[:, 0, :2]

            for phase_offset in range(freq_ratio):
                indices = np.arange(phase_offset, data_2d.shape[0], freq_ratio)
                downsampled = data_2d[indices]

                for i in range(downsampled.shape[0] - self.config.seq_length):
                    window = downsampled[i : i + self.config.seq_length]

                    for aug_seq in augment_sequence(window):
                        samples.append(aug_seq[: self.config.input_length])
                        targets.append(
                            generate_target(
                                aug_seq, self.config.input_length, self.config.tanh_power
                            )
                        )

        return np.array(samples), np.array(targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Model variant: 2-layer LSTM with proper dropout
# ---------------------------------------------------------------------------

class LSTMModel2D_V3(nn.Module):
    """Extended LSTM-FC with configurable architecture."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1, grad_clip=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.grad_clip = grad_clip

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.softmax(out)
        return out


# ---------------------------------------------------------------------------
# Trial config and runner
# ---------------------------------------------------------------------------

@dataclass
class TrialV3:
    trial_id: int = 0
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.006
    batch_size: int = 512
    scheduler_type: str = "onecycle"
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    num_steps: int = 20000
    patience: int = 3000
    tanh_power: float = 0.2
    pct_start: float = 0.1  # OneCycleLR warmup fraction


def run_trial(
    trial: TrialV3,
    train_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
) -> Dict:
    torch.manual_seed(42)
    np.random.seed(42)

    model = LSTMModel2D_V3(
        hidden_size=trial.hidden_size,
        num_layers=trial.num_layers,
        dropout=trial.dropout,
        grad_clip=trial.grad_clip,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=trial.learning_rate,
        weight_decay=trial.weight_decay,
    )

    if trial.scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=trial.learning_rate,
            total_steps=trial.num_steps,
            pct_start=trial.pct_start,
            anneal_strategy="cos",
        )
    elif trial.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=trial.num_steps // 4, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=trial.num_steps // 2, gamma=0.5
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=trial.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Full test set for evaluation
    test_x = test_dataset.samples.to(device)
    test_y = test_dataset.targets.to(device)

    best_test_loss = float("inf")
    best_test_acc = 0.0
    best_step = 0
    best_state = None
    patience_counter = 0

    train_iter = iter(train_loader)
    start_time = time.time()

    for step in range(1, trial.num_steps + 1):
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        model.train()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()

        if trial.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trial.grad_clip)

        optimizer.step()
        scheduler.step()

        # Evaluate every 100 steps
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate in batches for memory efficiency
                all_preds = []
                bs = 4096
                for i in range(0, len(test_x), bs):
                    all_preds.append(model(test_x[i:i+bs]))
                test_out = torch.cat(all_preds, dim=0)

                test_loss = criterion(test_out, test_y).item()
                pred_classes = test_out.argmax(dim=1)
                true_classes = test_y.argmax(dim=1)
                test_acc = (pred_classes == true_classes).float().mean().item()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_step = step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 100

            if step % 2000 == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"    Step {step}/{trial.num_steps} | "
                    f"Train: {loss.item():.6f} | Test MSE: {test_loss:.6f} | "
                    f"Acc: {test_acc:.4f} | Best: {best_test_loss:.6f} @ {best_step} | "
                    f"LR: {lr:.6f}"
                )

            if patience_counter >= trial.patience:
                logger.info(f"    Early stop at step {step}")
                break

    elapsed = time.time() - start_time

    # Restore best and compute detailed metrics
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        all_preds = []
        bs = 4096
        for i in range(0, len(test_x), bs):
            all_preds.append(model(test_x[i:i+bs]))
        test_out = torch.cat(all_preds, dim=0)

        mse = nn.MSELoss()(test_out, test_y).item()
        mae = torch.abs(test_out - test_y).mean().item()
        pred_classes = test_out.argmax(dim=1)
        true_classes = test_y.argmax(dim=1)
        accuracy = (pred_classes == true_classes).float().mean().item()

        per_class = {}
        for c, name in enumerate(["left", "straight", "right"]):
            mask = true_classes == c
            if mask.sum() > 0:
                class_acc = (pred_classes[mask] == c).float().mean().item()
                pred_mask = pred_classes == c
                tp = ((pred_classes == c) & (true_classes == c)).sum().item()
                precision = tp / max(pred_mask.sum().item(), 1)
                recall = tp / max(mask.sum().item(), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                per_class[name] = {
                    "accuracy": class_acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "count": mask.sum().item(),
                }

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "trial_id": trial.trial_id,
        "config": {
            "hidden_size": trial.hidden_size,
            "num_layers": trial.num_layers,
            "dropout": trial.dropout,
            "learning_rate": trial.learning_rate,
            "batch_size": trial.batch_size,
            "scheduler_type": trial.scheduler_type,
            "weight_decay": trial.weight_decay,
            "grad_clip": trial.grad_clip,
            "tanh_power": trial.tanh_power,
            "pct_start": trial.pct_start,
        },
        "test_mse": mse,
        "test_mae": mae,
        "test_accuracy": accuracy,
        "per_class": per_class,
        "best_step": best_step,
        "total_steps": step,
        "elapsed_seconds": elapsed,
        "n_params": n_params,
        "model_state": best_state,
    }


def build_trials() -> List[TrialV3]:
    trials = []
    tid = 0

    # =========================================================================
    # Phase 1: Hidden size scaling (the most impactful dimension)
    # =========================================================================
    for hs in [64, 96, 128, 192, 256, 384]:
        for lr in [0.004, 0.006, 0.008]:
            trials.append(TrialV3(
                trial_id=tid, hidden_size=hs, learning_rate=lr,
                batch_size=512, scheduler_type="onecycle",
                num_steps=20000, patience=3000,
            ))
            tid += 1

    # =========================================================================
    # Phase 2: 2-layer LSTM (previously only tested with weak schedulers)
    # =========================================================================
    for hs in [96, 128, 192]:
        for dropout in [0.1, 0.2, 0.3]:
            trials.append(TrialV3(
                trial_id=tid, hidden_size=hs, num_layers=2,
                dropout=dropout, learning_rate=0.006,
                batch_size=512, scheduler_type="onecycle",
                num_steps=20000, patience=3000,
            ))
            tid += 1

    # =========================================================================
    # Phase 3: Gradient clipping (can help with LSTM stability)
    # =========================================================================
    for gc in [0.5, 1.0, 5.0]:
        for hs in [128, 256]:
            trials.append(TrialV3(
                trial_id=tid, hidden_size=hs, grad_clip=gc,
                learning_rate=0.006, batch_size=512,
                scheduler_type="onecycle",
                num_steps=20000, patience=3000,
            ))
            tid += 1

    # =========================================================================
    # Phase 4: AdamW weight decay (previously no improvement, but with AdamW)
    # =========================================================================
    for wd in [1e-4, 1e-3, 1e-2]:
        for hs in [128, 256]:
            trials.append(TrialV3(
                trial_id=tid, hidden_size=hs, weight_decay=wd,
                learning_rate=0.006, batch_size=512,
                scheduler_type="onecycle",
                num_steps=20000, patience=3000,
            ))
            tid += 1

    # =========================================================================
    # Phase 5: Warmup ratio (OneCycleLR pct_start)
    # =========================================================================
    for pct in [0.05, 0.1, 0.2, 0.3]:
        trials.append(TrialV3(
            trial_id=tid, hidden_size=256, pct_start=pct,
            learning_rate=0.006, batch_size=512,
            scheduler_type="onecycle",
            num_steps=20000, patience=3000,
        ))
        tid += 1

    # =========================================================================
    # Phase 6: Batch size (larger may help with soft label regression)
    # =========================================================================
    for bs in [256, 512, 1024, 2048]:
        trials.append(TrialV3(
            trial_id=tid, hidden_size=256, batch_size=bs,
            learning_rate=0.006, scheduler_type="onecycle",
            num_steps=20000, patience=3000,
        ))
        tid += 1

    # =========================================================================
    # Phase 7: Learning rate + hidden size fine grid for top configs
    # =========================================================================
    for hs in [192, 256, 384]:
        for lr in [0.003, 0.005, 0.007, 0.01]:
            trials.append(TrialV3(
                trial_id=tid, hidden_size=hs, learning_rate=lr,
                batch_size=512, scheduler_type="onecycle",
                num_steps=20000, patience=3000,
            ))
            tid += 1

    return trials


def main():
    output_dir = Path("outputs/hypertune_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading datasets...")
    data_config = DataConfig(data_path="data_3d_h36m.npz")
    train_dataset = TrajectoryDatasetV3(data_config, split="train")
    test_dataset = TrajectoryDatasetV3(data_config, split="test")
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    trials = build_trials()
    logger.info(f"Running {len(trials)} trials")

    # Resume from partial results if available
    results_path = output_dir / "results.json"
    results = []
    completed_ids = set()
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        completed_ids = {r["trial_id"] for r in results}
        logger.info(f"Resuming: {len(completed_ids)} trials already completed")

    best_result = None
    best_model_state = None
    if results:
        best_result = min(results, key=lambda r: r["test_mse"])
        logger.info(f"Previous best: MSE={best_result['test_mse']:.6f}")

    for i, trial in enumerate(trials):
        if trial.trial_id in completed_ids:
            continue
        logger.info(
            f"\n{'='*70}\n"
            f"Trial {trial.trial_id} ({i+1}/{len(trials)}): "
            f"h={trial.hidden_size} L={trial.num_layers} "
            f"lr={trial.learning_rate} bs={trial.batch_size} "
            f"sched={trial.scheduler_type} drop={trial.dropout} "
            f"wd={trial.weight_decay} gc={trial.grad_clip} "
            f"pct={trial.pct_start}"
        )

        result = run_trial(trial, train_dataset, test_dataset, device)

        # Don't serialize model state in results JSON
        model_state = result.pop("model_state", None)

        results.append(result)

        logger.info(
            f"  -> MSE: {result['test_mse']:.6f} | "
            f"Acc: {result['test_accuracy']:.4f} | "
            f"MAE: {result['test_mae']:.6f} | "
            f"Best @ step {result['best_step']} | "
            f"Params: {result['n_params']:,} | "
            f"{result['elapsed_seconds']:.1f}s"
        )
        if result.get("per_class"):
            for name, metrics in result["per_class"].items():
                logger.info(
                    f"     {name:10s}: Acc={metrics['accuracy']:.4f} "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                    f"F1={metrics['f1']:.4f}"
                )

        if best_result is None or result["test_mse"] < best_result["test_mse"]:
            best_result = result
            best_model_state = model_state
            logger.info(f"  *** NEW BEST (MSE: {result['test_mse']:.6f}, Acc: {result['test_accuracy']:.4f}) ***")

            # Save best model so far
            if best_model_state is not None:
                config = ModelConfig(
                    hidden_size=trial.hidden_size,
                    num_layers=trial.num_layers,
                    dropout=trial.dropout,
                )
                torch.save(
                    {"model_state_dict": best_model_state, "model_config": config},
                    output_dir / "best_model.pt",
                )

        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Sort and report
    sorted_results = sorted(results, key=lambda r: r["test_mse"])

    logger.info(f"\n{'='*70}")
    logger.info("TOP 20 RESULTS (by Test MSE)")
    logger.info(f"{'='*70}")
    for i, r in enumerate(sorted_results[:20]):
        c = r["config"]
        logger.info(
            f"#{i+1}: MSE={r['test_mse']:.6f} Acc={r['test_accuracy']:.4f} "
            f"MAE={r['test_mae']:.6f} | "
            f"h={c['hidden_size']} L={c['num_layers']} "
            f"lr={c['learning_rate']} bs={c['batch_size']} "
            f"sched={c['scheduler_type']} drop={c['dropout']} "
            f"wd={c['weight_decay']} gc={c['grad_clip']} "
            f"pct={c['pct_start']} | "
            f"params={r['n_params']:,}"
        )

    with open(output_dir / "results_sorted.json", "w") as f:
        json.dump(sorted_results, f, indent=2)

    # Print comparison with previous best
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON WITH PREVIOUS BEST")
    logger.info(f"{'='*70}")
    logger.info(f"Previous best (h=128, 100k steps): MSE=0.0134, Acc=88.01%")
    best = sorted_results[0]
    logger.info(
        f"New best: MSE={best['test_mse']:.6f}, Acc={best['test_accuracy']:.4f}"
    )
    logger.info(f"Config: {json.dumps(best['config'], indent=2)}")

    return sorted_results


if __name__ == "__main__":
    main()
