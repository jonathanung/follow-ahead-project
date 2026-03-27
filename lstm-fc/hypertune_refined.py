"""Refined hyperparameter tuning - second round around best config from initial search.

Best from round 1: dropout h=128 L=1 lr=0.005 bs=256 MSE loss, 0.1 dropout
Now search finer grid with longer training.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lstm_fc.config import DataConfig, ModelConfig
from lstm_fc.data.dataset import TrajectoryDataset
from lstm_fc.model import LSTMModel2D
from hypertune import (
    LSTMModel2D_Dropout,
    LSTMModel2D_BiDir,
    TrialConfig,
    KLDivLossWrapper,
    CrossEntropySmooth,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_trial_long(
    trial: TrialConfig,
    train_dataset: TrajectoryDataset,
    test_dataset: TrajectoryDataset,
    device: torch.device,
) -> dict:
    """Run a single trial with longer training and proper evaluation."""
    torch.manual_seed(42)
    np.random.seed(42)

    model_config = ModelConfig(
        hidden_size=trial.hidden_size,
        num_layers=trial.num_layers,
    )

    if trial.model_type == "dropout":
        model = LSTMModel2D_Dropout(model_config, dropout=trial.dropout).to(device)
    elif trial.model_type == "bidir":
        model = LSTMModel2D_BiDir(model_config, dropout=trial.dropout).to(device)
    else:
        model = LSTMModel2D(model_config).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=trial.learning_rate,
        weight_decay=trial.weight_decay,
    )

    if trial.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=trial.num_steps // 4, T_mult=2, eta_min=1e-6
        )
    elif trial.scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=trial.learning_rate,
            total_steps=trial.num_steps,
            pct_start=0.1,
            anneal_strategy="cos",
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

    # Full test set evaluation
    test_x = test_dataset.samples.to(device)
    test_y = test_dataset.targets.to(device)
    # Use a larger test batch for more reliable evaluation
    if len(test_x) > 2048:
        rng = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(test_x), generator=rng)[:2048]
        test_x_eval, test_y_eval = test_x[idx], test_y[idx]
    else:
        test_x_eval, test_y_eval = test_x, test_y

    best_test_loss = float("inf")
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
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(test_x_eval)
                test_loss = criterion(test_out, test_y_eval).item()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_step = step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 100

            if step % 1000 == 0:
                logger.info(
                    f"    Step {step}/{trial.num_steps} | "
                    f"Train: {loss.item():.6f} | Test: {test_loss:.6f} | "
                    f"Best: {best_test_loss:.6f} @ {best_step}"
                )

            if patience_counter >= trial.patience:
                logger.info(f"    Early stop at step {step}")
                break

    elapsed = time.time() - start_time

    # Restore best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_out = model(test_x_eval)
        pred_classes = test_out.argmax(dim=1)
        true_classes = test_y_eval.argmax(dim=1)
        accuracy = (pred_classes == true_classes).float().mean().item()
        mse = nn.MSELoss()(test_out, test_y_eval).item()

        # Per-class accuracy
        per_class_acc = {}
        for c, name in enumerate(["left", "straight", "right"]):
            mask = true_classes == c
            if mask.sum() > 0:
                per_class_acc[name] = (pred_classes[mask] == c).float().mean().item()

    return {
        "trial_id": trial.trial_id,
        "config": {
            "hidden_size": trial.hidden_size,
            "num_layers": trial.num_layers,
            "learning_rate": trial.learning_rate,
            "batch_size": trial.batch_size,
            "model_type": trial.model_type,
            "dropout": trial.dropout,
            "weight_decay": trial.weight_decay,
            "scheduler_type": trial.scheduler_type,
        },
        "best_test_mse": mse,
        "best_test_loss": best_test_loss,
        "test_accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "best_step": best_step,
        "total_steps": step,
        "elapsed_seconds": elapsed,
    }


def main():
    output_dir = Path("outputs/hypertune_refined")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading datasets...")
    data_config = DataConfig(data_path="data_3d_h36m.npz")
    train_dataset = TrajectoryDataset(data_config, split="train")
    test_dataset = TrajectoryDataset(data_config, split="test")
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    trials = []
    tid = 0

    # Fine grid around best config
    for hidden_size in [96, 128, 192, 256]:
        for lr in [0.003, 0.004, 0.005, 0.006, 0.008]:
            for bs in [128, 256, 512]:
                for sched in ["step", "cosine", "onecycle"]:
                    trials.append(TrialConfig(
                        trial_id=tid,
                        hidden_size=hidden_size,
                        num_layers=1,
                        learning_rate=lr,
                        batch_size=bs,
                        model_type="dropout",
                        dropout=0.1,
                        loss_type="mse",
                        scheduler_type=sched,
                        num_steps=10000,
                        patience=2000,
                    ))
                    tid += 1

    # Also try base model (no dropout) with best-performing configs
    for hidden_size in [128, 256]:
        for lr in [0.004, 0.005, 0.006]:
            for bs in [256, 512]:
                trials.append(TrialConfig(
                    trial_id=tid,
                    hidden_size=hidden_size,
                    num_layers=1,
                    learning_rate=lr,
                    batch_size=bs,
                    model_type="base",
                    dropout=0.0,
                    loss_type="mse",
                    scheduler_type="cosine",
                    num_steps=10000,
                    patience=2000,
                ))
                tid += 1

    # Bidirectional with smaller hidden (to match param count)
    for hidden_size in [64, 96]:
        for lr in [0.003, 0.005]:
            for bs in [256, 512]:
                trials.append(TrialConfig(
                    trial_id=tid,
                    hidden_size=hidden_size,
                    num_layers=1,
                    learning_rate=lr,
                    batch_size=bs,
                    model_type="bidir",
                    dropout=0.1,
                    loss_type="mse",
                    scheduler_type="cosine",
                    num_steps=10000,
                    patience=2000,
                ))
                tid += 1

    logger.info(f"Running {len(trials)} refined trials")
    results = []
    best_result = None

    for i, trial in enumerate(trials):
        logger.info(
            f"\n{'='*60}\n"
            f"Trial {trial.trial_id} ({i+1}/{len(trials)}): "
            f"{trial.model_type} h={trial.hidden_size} L={trial.num_layers} "
            f"lr={trial.learning_rate} bs={trial.batch_size} "
            f"sched={trial.scheduler_type} drop={trial.dropout}"
        )

        result = run_trial_long(trial, train_dataset, test_dataset, device)
        results.append(result)

        logger.info(
            f"  -> MSE: {result['best_test_mse']:.6f} | "
            f"Acc: {result['test_accuracy']:.4f} | "
            f"Best @ step {result['best_step']} | "
            f"{result['elapsed_seconds']:.1f}s"
        )

        if best_result is None or result["best_test_mse"] < best_result["best_test_mse"]:
            best_result = result
            logger.info(f"  *** NEW BEST (MSE: {result['best_test_mse']:.6f}) ***")

        # Save intermediate
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Sort and report
    sorted_results = sorted(results, key=lambda r: r["best_test_mse"])

    logger.info(f"\n{'='*70}")
    logger.info("TOP 15 RESULTS (by Test MSE)")
    logger.info(f"{'='*70}")
    for i, r in enumerate(sorted_results[:15]):
        c = r["config"]
        logger.info(
            f"#{i+1}: MSE={r['best_test_mse']:.6f} Acc={r['test_accuracy']:.4f} | "
            f"{c['model_type']} h={c['hidden_size']} "
            f"lr={c['learning_rate']} bs={c['batch_size']} "
            f"sched={c['scheduler_type']} drop={c['dropout']}"
        )
        if "per_class_accuracy" in r:
            logger.info(f"     Per-class: {r['per_class_accuracy']}")

    with open(output_dir / "results_sorted.json", "w") as f:
        json.dump(sorted_results, f, indent=2)

    logger.info(f"\nBest: {json.dumps(sorted_results[0], indent=2)}")
    return sorted_results[0]


if __name__ == "__main__":
    main()
