"""
RL_interface.py — Value Function Bridge for MCTS Integration.

Ports RL_interface.py from Follow_ahead_reaction-ROS1/follow/scripts/,
keeping only the A2C path and updating it for the current SB3 API.

HOW IT IS USED IN MCTS
-----------------------
In search.py, the evaluate_node() function calls:

    obs = np.concatenate([state[0,:2] - state[1,:2], [state[0,2]-state[1,2]]])
    value = self.params['RL_model'].evaluate_state(obs, policy='a2c')

Our updated evaluate_state() accepts a flat (4,) numpy array and returns
the critic's V(s) estimate as a plain Python float — exactly what the MCTS
backpropagation step needs.

LOADING THE MODEL
-----------------
After training completes, load the exported policy:

    from RL_sim.RL_interface import RL_model
    rl = RL_model()
    rl.load_model('./models/a2c_follow_ahead')
    value = rl.evaluate_state(obs_array)   # obs_array: np.ndarray shape (4,)
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

    def load_model(self, model_path: str) -> None:
        """
        Load a trained A2C model from a .zip file saved by SB3.

        Parameters
        ----------
        model_path : str — path without the .zip extension, e.g.
                           './models/a2c_follow_ahead'
        """
        self.model = A2C.load(model_path)
        self.model.policy.set_training_mode(False)  # eval mode — no dropout etc.

    # ------------------------------------------------------------------
    # Value function query (called by MCTS search.py)
    # ------------------------------------------------------------------

    def evaluate_state(self, obs: np.ndarray) -> float:
        """
        Query the critic network to get V(s) for a given observation.

        This is called from search.py's evaluate_node() for every MCTS node
        to score how good the robot–human configuration is.

        Parameters
        ----------
        obs : np.ndarray — flat observation vector, shape (4,) float32
                           [dx, dy, human_theta, robot_theta]
                           (same format as FollowState.to_numpy())

        Returns
        -------
        float — scalar value estimate V(s)
        """
        assert self.model is not None, "Call load_model() before evaluate_state()."

        # Ensure correct dtype and shape: (1, 4) for the policy forward pass
        obs_t = torch.as_tensor(
            obs.astype(np.float32), dtype=torch.float32
        ).unsqueeze(0)  # shape: (1, 4)

        with torch.no_grad():
            # predict_values() is SB3's built-in critic call, returns (1,1) tensor
            value_t = self.model.policy.predict_values(obs_t)

        return float(value_t.item())

    # ------------------------------------------------------------------
    # Convenience: predict action (for eval / deployment)
    # ------------------------------------------------------------------

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict the best action for a given observation using the actor.

        Not used by MCTS (which selects actions via tree search), but useful
        for running the trained policy standalone on the real robot.

        Parameters
        ----------
        obs          : np.ndarray — shape (4,) float32
        deterministic: bool — if True, takes the argmax action; if False, samples

        Returns
        -------
        int — action index in [0, NUM_ACTIONS-1]
        """
        assert self.model is not None, "Call load_model() before predict_action()."
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)
