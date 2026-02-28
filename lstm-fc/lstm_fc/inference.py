"""Inference API for human action prediction.

This module provides the public interface that the MCTS layer imports.

Usage:
    from lstm_fc import HumanActionPredictor

    predictor = HumanActionPredictor("path/to/model_final.pt")
    probs = predictor.predict(trajectory_history)
    # probs == {"left": 0.15, "straight": 0.72, "right": 0.13}
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

from .config import ModelConfig
from .model import LSTMModel2D


class HumanActionPredictor:
    """Predicts human action probabilities from 2D trajectory history.

    Loads a trained LSTM-FC model and provides inference. Designed as
    the integration point for the MCTS layer.

    Args:
        model_path: Path to a saved model checkpoint (.pt file).
        device: Device string ("cuda", "cpu", or "auto").
    """

    ACTION_NAMES = ["left", "straight", "right"]

    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
        else:
            config = ModelConfig()

        self.model = LSTMModel2D(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self, history: Union[List[List[float]], np.ndarray]
    ) -> Dict[str, float]:
        """Predict action probabilities from trajectory history.

        Normalizes by subtracting the last point so the model sees
        displacement patterns relative to the current position.

        Args:
            history: Trajectory of 2D points, shape (N, 2) where N is
                     the input_length (default 14). Each point is [x, y].

        Returns:
            Dict with keys "left", "straight", "right" mapping to
            float probabilities summing to ~1.0.
        """
        if isinstance(history, list):
            history = np.array(history, dtype=np.float32)

        tensor = torch.tensor(history, dtype=torch.float32, device=self.device)

        # Normalize: subtract last point
        tensor = tensor - tensor[-1].clone()

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        output = self.model(tensor).squeeze(0).cpu().numpy()

        return {
            name: float(output[i]) for i, name in enumerate(self.ACTION_NAMES)
        }

    @torch.no_grad()
    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        """Batch prediction for efficiency.

        Args:
            histories: Array of shape (batch, seq_len, 2). Caller is
                       responsible for normalization.

        Returns:
            Array of shape (batch, 3) with [left, straight, right] probabilities.
        """
        tensor = torch.tensor(histories, dtype=torch.float32, device=self.device)
        return self.model(tensor).cpu().numpy()
