"""Hyperparameter tuning for LSTM-FC model.

Performs systematic search over key hyperparameters with early stopping,
and evaluates multiple loss function and architecture variants.
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lstm_fc.config import DataConfig, ModelConfig, TrainingConfig
from lstm_fc.data.dataset import TrajectoryDataset
from lstm_fc.model import LSTMModel2D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended model variants
# ---------------------------------------------------------------------------

class LSTMModel2D_Dropout(nn.Module):
    """LSTM-FC with dropout regularization."""

    def __init__(self, config: ModelConfig, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.softmax(out)
        return out


class LSTMModel2D_BiDir(nn.Module):
    """Bidirectional LSTM-FC."""

    def __init__(self, config: ModelConfig, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.softmax(out)
        return out


class LSTMModel2D_Deep(nn.Module):
    """LSTM-FC with an extra hidden FC layer."""

    def __init__(self, config: ModelConfig, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        fc_hidden = config.hidden_size // 2
        self.fc1 = nn.Linear(config.hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, config.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.softmax(out)
        return out


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class KLDivLossWrapper(nn.Module):
    """KL Divergence loss for probability distributions."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_pred = torch.log(pred + 1e-8)
        return nn.functional.kl_div(log_pred, target, reduction="batchmean")


class CrossEntropySmooth(nn.Module):
    """Cross-entropy with soft labels."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_pred = torch.log(pred + 1e-8)
        return -(target * log_pred).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Trial configuration
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    """Configuration for a single hyperparameter trial."""
    trial_id: int = 0
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 0.01
    batch_size: int = 64
    model_type: str = "base"  # base, dropout, bidir, deep
    dropout: float = 0.1
    loss_type: str = "mse"  # mse, kl, ce
    weight_decay: float = 0.0
    scheduler_type: str = "step"  # step, cosine, plateau
    num_steps: int = 5000
    patience: int = 500  # Early stopping patience


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

def run_trial(
    trial: TrialConfig,
    train_dataset: TrajectoryDataset,
    test_dataset: TrajectoryDataset,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """Run a single hyperparameter trial with early stopping.

    Returns dict with trial results.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    model_config = ModelConfig(
        hidden_size=trial.hidden_size,
        num_layers=trial.num_layers,
    )

    # Select model architecture
    if trial.model_type == "dropout":
        model = LSTMModel2D_Dropout(model_config, dropout=trial.dropout).to(device)
    elif trial.model_type == "bidir":
        model = LSTMModel2D_BiDir(model_config, dropout=trial.dropout).to(device)
    elif trial.model_type == "deep":
        model = LSTMModel2D_Deep(model_config, dropout=trial.dropout).to(device)
    else:
        model = LSTMModel2D(model_config).to(device)

    # Select loss
    if trial.loss_type == "kl":
        criterion = KLDivLossWrapper()
    elif trial.loss_type == "ce":
        criterion = CrossEntropySmooth()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=trial.learning_rate,
        weight_decay=trial.weight_decay,
    )

    # Select scheduler
    if trial.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=trial.num_steps, eta_min=1e-6
        )
    elif trial.scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=200, factor=0.5
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=trial.num_steps, gamma=0.5
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=trial.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Pre-compute test batch
    test_x = test_dataset.samples.to(device)
    test_y = test_dataset.targets.to(device)
    if len(test_x) > 512:
        rng = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(test_x), generator=rng)[:512]
        test_x, test_y = test_x[idx], test_y[idx]

    # Training loop with early stopping
    best_test_loss = float("inf")
    best_step = 0
    patience_counter = 0
    train_losses = []
    test_losses = []

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
        optimizer.step()

        if trial.scheduler_type == "plateau":
            pass  # Updated after test loss
        else:
            scheduler.step()

        # Evaluate every 50 steps
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(test_x)
                test_loss = criterion(test_out, test_y).item()

            train_losses.append(loss.item())
            test_losses.append(test_loss)

            if trial.scheduler_type == "plateau":
                scheduler.step(test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_step = step
                patience_counter = 0
            else:
                patience_counter += 50

            if patience_counter >= trial.patience:
                logger.info(
                    f"  Trial {trial.trial_id}: Early stopping at step {step} "
                    f"(best={best_test_loss:.6f} at step {best_step})"
                )
                break

    elapsed = time.time() - start_time

    # Compute additional metrics on test set
    model.eval()
    with torch.no_grad():
        test_out = model(test_x)
        # Classification accuracy (argmax match)
        pred_classes = test_out.argmax(dim=1)
        true_classes = test_y.argmax(dim=1)
        accuracy = (pred_classes == true_classes).float().mean().item()
        # MSE regardless of training loss type
        mse = nn.MSELoss()(test_out, test_y).item()

    result = {
        "trial_id": trial.trial_id,
        "config": {
            "hidden_size": trial.hidden_size,
            "num_layers": trial.num_layers,
            "learning_rate": trial.learning_rate,
            "batch_size": trial.batch_size,
            "model_type": trial.model_type,
            "dropout": trial.dropout,
            "loss_type": trial.loss_type,
            "weight_decay": trial.weight_decay,
            "scheduler_type": trial.scheduler_type,
        },
        "best_test_loss": best_test_loss,
        "best_step": best_step,
        "final_train_loss": train_losses[-1] if train_losses else float("inf"),
        "final_test_loss": test_losses[-1] if test_losses else float("inf"),
        "test_mse": mse,
        "test_accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "total_steps": step,
    }

    return result


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def build_trials() -> List[TrialConfig]:
    """Build list of trials covering hyperparameter space."""
    trials = []
    tid = 0

    # Phase 1: Architecture + hidden size search (with base hyperparams)
    for model_type in ["base", "dropout", "bidir", "deep"]:
        for hidden_size in [32, 64, 128, 256]:
            trials.append(TrialConfig(
                trial_id=tid,
                hidden_size=hidden_size,
                num_layers=1,
                learning_rate=0.005,
                batch_size=64,
                model_type=model_type,
                dropout=0.1,
                loss_type="mse",
                num_steps=3000,
                patience=500,
            ))
            tid += 1

    # Phase 2: Loss function comparison (with promising architectures)
    for loss_type in ["mse", "kl", "ce"]:
        for hidden_size in [64, 128]:
            trials.append(TrialConfig(
                trial_id=tid,
                hidden_size=hidden_size,
                num_layers=1,
                learning_rate=0.005,
                batch_size=64,
                model_type="dropout",
                dropout=0.1,
                loss_type=loss_type,
                num_steps=3000,
                patience=500,
            ))
            tid += 1

    # Phase 3: Learning rate search
    for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
        trials.append(TrialConfig(
            trial_id=tid,
            hidden_size=128,
            num_layers=1,
            learning_rate=lr,
            batch_size=64,
            model_type="dropout",
            dropout=0.1,
            loss_type="mse",
            num_steps=3000,
            patience=500,
        ))
        tid += 1

    # Phase 4: Batch size search
    for bs in [32, 64, 128, 256]:
        trials.append(TrialConfig(
            trial_id=tid,
            hidden_size=128,
            num_layers=1,
            learning_rate=0.005,
            batch_size=bs,
            model_type="dropout",
            dropout=0.1,
            loss_type="mse",
            num_steps=3000,
            patience=500,
        ))
        tid += 1

    # Phase 5: Depth + scheduler exploration
    for num_layers in [1, 2, 3]:
        for sched in ["step", "cosine", "plateau"]:
            trials.append(TrialConfig(
                trial_id=tid,
                hidden_size=128,
                num_layers=num_layers,
                learning_rate=0.005,
                batch_size=64,
                model_type="dropout",
                dropout=0.15,
                loss_type="mse",
                scheduler_type=sched,
                num_steps=3000,
                patience=500,
            ))
            tid += 1

    # Phase 6: Dropout rates
    for dropout in [0.0, 0.05, 0.1, 0.2, 0.3]:
        trials.append(TrialConfig(
            trial_id=tid,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.005,
            batch_size=64,
            model_type="dropout",
            dropout=dropout,
            loss_type="mse",
            num_steps=3000,
            patience=500,
        ))
        tid += 1

    # Phase 7: Weight decay
    for wd in [0.0, 1e-5, 1e-4, 1e-3]:
        trials.append(TrialConfig(
            trial_id=tid,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.005,
            batch_size=64,
            model_type="dropout",
            dropout=0.1,
            loss_type="mse",
            weight_decay=wd,
            num_steps=3000,
            patience=500,
        ))
        tid += 1

    return trials


def main():
    output_dir = Path("outputs/hypertune")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data once
    logger.info("Loading datasets...")
    data_config = DataConfig(data_path="data_3d_h36m.npz")
    train_dataset = TrajectoryDataset(data_config, split="train")
    test_dataset = TrajectoryDataset(data_config, split="test")
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    trials = build_trials()
    logger.info(f"Running {len(trials)} trials")

    results = []
    best_result = None

    for i, trial in enumerate(trials):
        logger.info(
            f"\n{'='*60}\n"
            f"Trial {trial.trial_id} ({i+1}/{len(trials)}): "
            f"{trial.model_type} h={trial.hidden_size} L={trial.num_layers} "
            f"lr={trial.learning_rate} bs={trial.batch_size} "
            f"loss={trial.loss_type} sched={trial.scheduler_type} "
            f"drop={trial.dropout} wd={trial.weight_decay}"
        )

        result = run_trial(trial, train_dataset, test_dataset, device, output_dir)
        results.append(result)

        logger.info(
            f"  -> Test MSE: {result['test_mse']:.6f} | "
            f"Accuracy: {result['test_accuracy']:.4f} | "
            f"Best loss: {result['best_test_loss']:.6f} at step {result['best_step']} | "
            f"Time: {result['elapsed_seconds']:.1f}s"
        )

        if best_result is None or result["test_mse"] < best_result["test_mse"]:
            best_result = result
            logger.info(f"  *** NEW BEST (MSE: {result['test_mse']:.6f}) ***")

        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Sort by test MSE and print top results
    sorted_results = sorted(results, key=lambda r: r["test_mse"])

    logger.info(f"\n{'='*70}")
    logger.info("TOP 10 RESULTS (by Test MSE)")
    logger.info(f"{'='*70}")
    for i, r in enumerate(sorted_results[:10]):
        c = r["config"]
        logger.info(
            f"#{i+1}: MSE={r['test_mse']:.6f} Acc={r['test_accuracy']:.4f} | "
            f"{c['model_type']} h={c['hidden_size']} L={c['num_layers']} "
            f"lr={c['learning_rate']} bs={c['batch_size']} "
            f"loss={c['loss_type']} sched={c['scheduler_type']} "
            f"drop={c['dropout']} wd={c['weight_decay']}"
        )

    # Save final sorted results
    with open(output_dir / "results_sorted.json", "w") as f:
        json.dump(sorted_results, f, indent=2)

    logger.info(f"\nBest config: {json.dumps(sorted_results[0]['config'], indent=2)}")
    logger.info(f"Results saved to {output_dir}")

    return sorted_results[0]


if __name__ == "__main__":
    main()
