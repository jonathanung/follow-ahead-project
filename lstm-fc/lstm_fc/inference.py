"""Inference API for human action prediction.

This module is the public interface that the MCTS planner imports.
It provides two main classes:

* :class:`HumanActionPredictor` — loads a trained LSTM-FC checkpoint and
  returns action probabilities from a trajectory snippet.
* :class:`TrajectoryBuffer` — a fixed-length sliding window that
  accumulates 2-D positions at the decision frequency (5 Hz) and
  produces the input array the predictor expects.

Quick-start
-----------
::

    from lstm_fc import HumanActionPredictor, TrajectoryBuffer
    from lstm_fc.actions import INPUT_LENGTH, DECISION_HZ

    predictor = HumanActionPredictor("model_final.pt")
    buf = TrajectoryBuffer(length=INPUT_LENGTH)

    # In your 5 Hz control loop:
    buf.push(human_x, human_y)
    if buf.ready:
        probs = predictor.predict(buf.get())
        # probs == {"left": 0.12, "straight": 0.76, "right": 0.12}
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .actions import HUMAN_ACTION_NAMES, INPUT_LENGTH
from .config import ModelConfig
from .model import LSTMModel2D


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """Fixed-length sliding window of 2-D trajectory points.

    Call :meth:`push` once per decision step (5 Hz) with the latest
    human position.  When enough points have been collected the buffer
    reports :attr:`ready` ``= True`` and :meth:`get` returns the
    ``(length, 2)`` array the predictor expects.

    Parameters
    ----------
    length : int, optional
        Number of points the LSTM model expects.  Defaults to
        :data:`lstm_fc.actions.INPUT_LENGTH` (14).
    """

    def __init__(self, length: int = INPUT_LENGTH):
        self._length = length
        self._buf: deque = deque(maxlen=length)

    # -- mutators -----------------------------------------------------------

    def push(self, x: float, y: float) -> None:
        """Append a new position and discard the oldest if full."""
        self._buf.append((x, y))

    def clear(self) -> None:
        """Empty the buffer (e.g. on tracking loss)."""
        self._buf.clear()

    # -- queries ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """``True`` when the buffer contains ``length`` points."""
        return len(self._buf) == self._length

    @property
    def length(self) -> int:
        """Number of points expected."""
        return self._length

    @property
    def count(self) -> int:
        """Number of points currently stored."""
        return len(self._buf)

    def get(self) -> np.ndarray:
        """Return the current window as a ``(length, 2)`` float32 array.

        Raises
        ------
        RuntimeError
            If the buffer does not yet have ``length`` points.
        """
        if not self.ready:
            raise RuntimeError(
                f"TrajectoryBuffer has {len(self._buf)}/{self._length} points"
            )
        return np.array(self._buf, dtype=np.float32)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class HumanActionPredictor:
    """Predicts human action probabilities from 2-D trajectory history.

    Loads a trained LSTM-FC checkpoint and exposes :meth:`predict` (single
    trajectory) and :meth:`predict_batch` (multiple trajectories).

    The MCTS planner is the primary consumer.  It calls :meth:`predict`
    once per 5 Hz decision step and uses the returned dict to weight the
    UCB score of human-turn nodes::

        ucb = (child.value / child.n
               + C * sqrt(log(parent.n) / child.n)) * human_prob[action]

    Parameters
    ----------
    model_path : str or Path
        Path to a ``.pt`` checkpoint containing ``model_state_dict``
        and optionally ``model_config``.
    device : str, optional
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (default).  ``"auto"``
        selects CUDA when available.
    """

    ACTION_NAMES: Tuple[str, ...] = HUMAN_ACTION_NAMES

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
            config: ModelConfig = checkpoint["model_config"]
        else:
            config = ModelConfig()

        self._input_length = getattr(config, "_input_length", INPUT_LENGTH)
        self.model = LSTMModel2D(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, history: Union[List[List[float]], np.ndarray]
    ) -> Dict[str, float]:
        """Predict action probabilities from a single trajectory.

        The input is automatically normalized (last point subtracted)
        so the model sees displacement patterns only.

        Parameters
        ----------
        history : array-like, shape ``(N, 2)``
            Ordered 2-D positions ``[x, y]``.  ``N`` should equal
            :data:`~lstm_fc.actions.INPUT_LENGTH` (14).

        Returns
        -------
        dict
            ``{"left": p_l, "straight": p_s, "right": p_r}`` where
            values are floats in ``[0, 1]`` summing to ~1.0.
        """
        if isinstance(history, list):
            history = np.array(history, dtype=np.float32)

        tensor = torch.tensor(history, dtype=torch.float32, device=self.device)

        # Normalize: make positions relative to the last observed point.
        tensor = tensor - tensor[-1].clone()

        # (1, N, 2) → model → (1, 3) → squeeze
        output = self.model(tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

        return {
            name: float(output[i]) for i, name in enumerate(self.ACTION_NAMES)
        }

    @torch.no_grad()
    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        """Batch prediction with automatic normalization.

        Parameters
        ----------
        histories : ndarray, shape ``(B, N, 2)``
            ``B`` trajectories, each ``N`` points of ``[x, y]``.

        Returns
        -------
        ndarray, shape ``(B, 3)``
            ``[p_left, p_straight, p_right]`` per trajectory.
        """
        tensor = torch.tensor(histories, dtype=torch.float32, device=self.device)

        # Normalize each trajectory by its last point.
        last = tensor[:, -1:, :].clone()
        tensor = tensor - last

        return self.model(tensor).cpu().numpy()

    @torch.no_grad()
    def predict_raw(self, histories_normalized: np.ndarray) -> np.ndarray:
        """Batch prediction **without** normalization.

        Use this only if you have already subtracted the last point
        from each trajectory yourself (e.g. during evaluation on the
        training dataset where normalization was done at load time).

        Parameters
        ----------
        histories_normalized : ndarray, shape ``(B, N, 2)``
            Already-normalized trajectories.

        Returns
        -------
        ndarray, shape ``(B, 3)``
        """
        tensor = torch.tensor(
            histories_normalized, dtype=torch.float32, device=self.device
        )
        return self.model(tensor).cpu().numpy()
