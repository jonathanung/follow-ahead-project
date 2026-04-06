"""
RL_interface.py — Value Function Bridge for MCTS Integration.

Replaces the original follow/scripts/RL_interface.py which had three bugs:
  1. Called state.to(DEVICE) — assumed input was already a tensor (crashed on numpy)
  2. Hardcoded DEVICE = 'cuda' — crashed on CPU machines
  3. evaluate_state() returned a tensor, not a float

This version is the corrected wrapper from RL_sim/RL_interface.py, adapted for
use in planner.py and any other MCTS script.

HOW IT IS USED IN MCTS (planner.py)
--------------------------------------
    from RL_interface import RL_model
    rl = RL_model()
    rl.load_model('<repo>/RL_sim/models/a2c_follow_ahead')
    value = rl.evaluate_state(obs_array)   # obs_array: np.ndarray shape (4,) float32

INPUT
-----
    obs : np.ndarray shape (4,) float32
          [dx, dy, human_theta, robot_theta]
          dx, dy in metres (robot position relative to human)
          angles in radians

OUTPUT
------
    float — scalar V(s) from the A2C critic network
"""

import numpy as np
import torch
from stable_baselines3 import A2C


class RL_model:
    """
    Thin wrapper around a trained SB3 A2C model that exposes
    the critic value function for use in MCTS node evaluation.
    """

    def __init__(self) -> None:
        self.model: A2C | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: str, policy: str = 'a2c', env=None) -> None:
        """
        Load a trained A2C model from a .zip file saved by SB3.

        Parameters
        ----------
        model_path : str  — path without the .zip extension, e.g.
                            './models/a2c_follow_ahead'
        policy     : str  — only 'a2c' is supported (ignored, kept for API compat)
        env        : ignored (kept for API compat with old interface)
        """
        self.model = A2C.load(model_path)
        self.model.policy.set_training_mode(False)   # eval mode — no dropout

    # ------------------------------------------------------------------
    # Value function query (called by MCTS planner.py)
    # ------------------------------------------------------------------

    def evaluate_state(self, obs: np.ndarray, policy: str = 'a2c') -> float:
        """
        Query the critic network to get V(s) for a given observation.

        Parameters
        ----------
        obs    : np.ndarray — flat observation vector, shape (4,) float32
                              [dx, dy, human_theta, robot_theta]
        policy : str        — ignored (kept for API compat with old interface)

        Returns
        -------
        float — scalar value estimate V(s)
        """
        assert self.model is not None, "Call load_model() before evaluate_state()."

        # Accept both numpy arrays and torch tensors (old interface passed tensors)
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy().astype(np.float32).flatten()
        else:
            obs_np = np.asarray(obs, dtype=np.float32).flatten()

        # SB3 policy forward expects shape (1, 4)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            value_t = self.model.policy.predict_values(obs_t)

        return float(value_t.item())

    # ------------------------------------------------------------------
    # Convenience: predict action (for standalone eval / deployment)
    # ------------------------------------------------------------------

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict the best action for a given observation using the actor.

        Not used by MCTS (which selects actions via tree search), but useful
        for running the trained policy standalone.

        Parameters
        ----------
        obs           : np.ndarray — shape (4,) float32
        deterministic : bool       — if True, argmax; if False, samples

        Returns
        -------
        int — action index
        """
        assert self.model is not None, "Call load_model() before predict_action()."
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)