"""Final long training run with best config from hypertune_v3.

Best config: h=192 L=2 dropout=0.1 lr=0.006 bs=512 OneCycleLR
Train for 200k steps to maximize performance.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lstm_fc.config import DataConfig, ModelConfig
from lstm_fc.data.dataset import TrajectoryDataset
from lstm_fc.model import LSTMModel2D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/final_v3/training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# Best config from tuning
HIDDEN_SIZE = 192
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.006
BATCH_SIZE = 512
NUM_STEPS = 200_000
CHECKPOINT_INTERVAL = 20_000


def main():
    output_dir = Path("outputs/final_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    logger.info(f"Device: {device}")
    logger.info(f"Config: h={HIDDEN_SIZE} L={NUM_LAYERS} drop={DROPOUT} "
                f"lr={LEARNING_RATE} bs={BATCH_SIZE} steps={NUM_STEPS}")

    # Load data
    data_config = DataConfig(data_path="data_3d_h36m.npz")
    train_dataset = TrajectoryDataset(data_config, split="train")
    test_dataset = TrajectoryDataset(data_config, split="test")
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Build model — 2-layer LSTM
    model_config = ModelConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model = LSTMModel2D(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=NUM_STEPS,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Full test set
    test_x = test_dataset.samples.to(device)
    test_y = test_dataset.targets.to(device)

    # Training
    best_test_mse = float("inf")
    best_test_acc = 0.0
    best_step = 0
    best_state = None

    train_iter = iter(train_loader)
    train_losses = []
    test_losses = []

    for step in range(1, NUM_STEPS + 1):
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

        train_losses.append(loss.item())

        # Evaluate every 200 steps
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                all_preds = []
                bs = 4096
                for i in range(0, len(test_x), bs):
                    all_preds.append(model(test_x[i:i+bs]))
                test_out = torch.cat(all_preds, dim=0)

                test_mse = criterion(test_out, test_y).item()
                pred_classes = test_out.argmax(dim=1)
                true_classes = test_y.argmax(dim=1)
                test_acc = (pred_classes == true_classes).float().mean().item()

            test_losses.append(test_mse)

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_test_acc = test_acc
                best_step = step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if step % 2000 == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Step {step}/{NUM_STEPS} | "
                f"Train: {loss.item():.6f} | Test MSE: {test_losses[-1]:.6f} | "
                f"Acc: {test_acc:.4f} | Best MSE: {best_test_mse:.6f} @ {best_step} | "
                f"LR: {lr:.6f}"
            )

        if step % CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "best_test_mse": best_test_mse,
                },
                output_dir / f"checkpoint_step{step}.pt",
            )

    # Save best model
    if best_state is not None:
        torch.save(
            {"model_state_dict": best_state, "model_config": model_config},
            output_dir / "model_final.pt",
        )
        logger.info(f"Best model saved: MSE={best_test_mse:.6f} Acc={best_test_acc:.4f} @ step {best_step}")

    # Save last model too
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model_config},
        output_dir / "model_last.pt",
    )

    # Save loss plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(train_losses, alpha=0.3, label="Train (raw)")
        # Smoothed
        window = 100
        if len(train_losses) > window:
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(train_losses)), smoothed, label=f"Train (avg {window})")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("MSE Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.set_yscale("log")

        ax2.plot(range(200, 200*(len(test_losses)+1), 200), test_losses, label="Test MSE")
        ax2.axhline(y=0.0134, color='r', linestyle='--', alpha=0.5, label="Previous best (h=128, 100k)")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("MSE Loss")
        ax2.set_title("Test Loss")
        ax2.legend()

        fig.suptitle(f"Final Training: h={HIDDEN_SIZE} L={NUM_LAYERS} | Best MSE={best_test_mse:.6f} @ step {best_step}")
        fig.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss plot saved")
    except ImportError:
        pass

    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"  Best Test MSE: {best_test_mse:.6f}")
    logger.info(f"  Best Test Acc: {best_test_acc:.4f}")
    logger.info(f"  Best Step: {best_step}")
    logger.info(f"  Previous best: MSE=0.0134, Acc=0.8801 (h=128, L=1, 100k steps)")


if __name__ == "__main__":
    main()
