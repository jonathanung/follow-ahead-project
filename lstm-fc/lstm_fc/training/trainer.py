import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import ModelConfig, TrainingConfig, DataConfig
from ..model import LSTMModel2D
from ..data.dataset import TrajectoryDataset

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training, evaluation, checkpointing, and logging.

    Args:
        model_config: Architecture parameters.
        training_config: Training hyperparameters.
        data_config: Data loading parameters.
    """

    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        training_config: TrainingConfig = TrainingConfig(),
        data_config: DataConfig = DataConfig(),
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config

        # Resolve device
        if training_config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(training_config.device)

        # Seed for reproducibility
        torch.manual_seed(training_config.seed)
        np.random.seed(training_config.seed)

        # Build model
        self.model = LSTMModel2D(model_config).to(self.device)

        # Loss, optimizer, scheduler
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=training_config.learning_rate
        )
        sched_type = getattr(training_config, "scheduler_type", "step")
        if sched_type == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=training_config.learning_rate,
                total_steps=training_config.num_epochs,
                pct_start=0.1,
                anneal_strategy="cos",
            )
        elif sched_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=training_config.num_epochs // 4,
                T_mult=2,
                eta_min=1e-6,
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.scheduler_step,
                gamma=training_config.scheduler_gamma,
            )

        # Load data
        self.train_dataset = TrajectoryDataset(data_config, split="train")
        self.test_dataset = TrajectoryDataset(data_config, split="test")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Output directory
        self.output_dir = Path(training_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss history
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []

    def train(self) -> Path:
        """Run the full training loop.

        Uses a step-based loop matching the reference semantics: each step
        draws one batch. The 100k epoch count maps to 100k steps.

        Returns:
            Path to the saved model checkpoint.
        """
        logger.info(
            f"Training on {self.device} for {self.training_config.num_epochs} steps"
        )
        logger.info(
            f"Train samples: {len(self.train_dataset)}, "
            f"Test samples: {len(self.test_dataset)}"
        )

        train_iter = iter(self.train_loader)

        for step in range(1, self.training_config.num_epochs + 1):
            # Get next batch, cycling through dataset
            try:
                batch_x, batch_y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch_x, batch_y = next(train_iter)

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.model.train()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.train_losses.append(loss.item())

            # Evaluate on test set
            test_loss = self._evaluate_step()
            self.test_losses.append(test_loss)

            # Logging
            if step % self.training_config.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {step}/{self.training_config.num_epochs} | "
                    f"Train Loss: {loss.item():.6f} | "
                    f"Test Loss: {test_loss:.6f} | LR: {lr:.6f}"
                )

            # Checkpointing
            if step % self.training_config.checkpoint_interval == 0:
                self._save_checkpoint(step)

        # Final save
        model_path = self._save_final()
        self._save_loss_plot()

        return model_path

    @torch.no_grad()
    def _evaluate_step(self) -> float:
        self.model.eval()
        batch_x = self.test_dataset.samples.to(self.device)
        batch_y = self.test_dataset.targets.to(self.device)

        if len(batch_x) > self.training_config.batch_size:
            idx = torch.randperm(len(batch_x))[: self.training_config.batch_size]
            batch_x, batch_y = batch_x[idx], batch_y[idx]

        outputs = self.model(batch_x)
        return self.criterion(outputs, batch_y).item()

    def _save_checkpoint(self, step: int) -> None:
        path = self.output_dir / f"checkpoint_step{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": self.model_config,
                "train_loss": self.train_losses[-1],
                "test_loss": self.test_losses[-1],
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def _save_final(self) -> Path:
        path = self.output_dir / "model_final.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model_config,
            },
            path,
        )
        logger.info(f"Final model saved: {path}")
        return path

    def _save_loss_plot(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_losses, label="Train Loss", alpha=0.7)
            ax.plot(self.test_losses, label="Test Loss", alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss (MSE)")
            ax.set_title("Training Progress")
            ax.legend()
            ax.set_yscale("log")
            fig.savefig(
                self.output_dir / "loss_curve.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
            logger.info("Loss plot saved")
        except ImportError:
            logger.warning("matplotlib not available, skipping loss plot")
