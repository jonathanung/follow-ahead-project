"""
train_a2c.py — A2C Training Entry Point for Follow-Ahead RL
============================================================

PURPOSE
-------
This script trains an Advantage Actor-Critic (A2C) policy to control a robot
that must follow *ahead* of a walking human in 2D space.

WHAT WE ARE TRAINING
--------------------
The agent (robot) observes a 4-dimensional state vector each timestep:
    obs = [dx, dy, human_theta, agent_theta]
      dx, dy        : robot position relative to the human [grid units]
      human_theta   : global heading of the human [rad]
      agent_theta   : global heading of the robot  [rad]

It must learn to select one of **6 discrete actions** (left/right/straight × 2 speeds)
to always stay ~1.5 m *directly ahead* of the human — i.e., in the forward-facing
cone — as the human performs a random walk.

The reward signal comes from nav_env._calculate_reward(), which combines:
    r_d    (Eq. 3a) — distance reward: peaked at d=1.5, penalizes too close / too far
    r_o    (Eq. 3b) — orientation reward: rewards being within a 25° cone ahead
    r_total (Eq. 3c) = clip(r_d + r_o, -1, 1)

WHY A2C?
--------
A2C (Synchronous Advantage Actor-Critic) is chosen because:
- It is on-policy, meaning the policy we train is the policy we evaluate —
  important for a real-time robot system.
- It is simpler and more stable than PPO for initial prototyping.
- Stable-Baselines3's A2C is fully tested with Gymnasium 1.x, which is what
  nav_env.py targets.

HOW TO RUN
----------
    # Activate the virtual environment first
    source ../rl_sim_env/bin/activate

    # Run with defaults (50 000 steps, logs to ./logs/)
    python train_a2c.py

    # Run with more steps and a custom log directory
    python train_a2c.py --total-timesteps 500000 --log-dir ./longer_run/

OUTPUTS
-------
    logs/A2C_<n>/        — TensorBoard event files; view with:
                              tensorboard --logdir logs/
    models/a2c_follow_ahead.zip  — Final saved policy (reload with A2C.load())
"""

import argparse
import os
import torch

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import our custom 2-D simulation environment
from nav_env import Environment   # defined in RL_sim/nav_env.py


# ---------------------------------------------------------------------------
# Hyperparameters
# (Kept as module-level constants so they are easy to spot and change.)
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS   = 1_000_000  # 1M steps — needed for tight 6°/step kinematics to converge
N_ENVS            = 4          # Number of parallel envs for faster rollout collection
POLICY            = "MlpPolicy" # Standard MLP actor-critic (no CNN needed for 4-dim obs)
LEARNING_RATE     = 7e-4       # SB3's A2C default
N_STEPS           = 20         # Matches MCTS MAX_DEPTH — V(s) trained on the same horizon MCTS evaluates
GAMMA             = 0.99       # Discount factor
ENT_COEF          = 0.01       # Entropy bonus — encourages exploration of the action space
LOG_DIR           = "./logs/"
CHECKPOINT_DIR    = "./checkpoints/"
FINAL_MODEL_PATH  = "./models/a2c_follow_ahead"


# ---------------------------------------------------------------------------
# Argument parsing — lets you override key params from the command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an A2C agent on the Follow-Ahead 2-D environment."
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total environment steps to train for (default: {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--n-envs", type=int, default=N_ENVS,
        help=f"Number of parallel environments (default: {N_ENVS})"
    )
    parser.add_argument(
        "--log-dir", type=str, default=LOG_DIR,
        help=f"TensorBoard log directory (default: {LOG_DIR})"
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Disable evaluation callback during training"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env() -> gym.Env:
    """
    Wraps our custom Environment so SB3's VecEnv can create multiple instances.

    Parameters match the corrected environment:
        world_size = 20.0  — 20m × 20m world (realistic indoor/outdoor scale)
        max_steps  = 200   — longer episodes give the agent time to converge
    Returns a single gym.Env instance.
    """
    # Kinematics match MCTS planner constants (planner.py) exactly.
    # world_size=10m: enough for follow-ahead geometry at 0.10 m/step.
    return Environment(
        robot_vel=0.10,
        robot_vel_fast=0.12,
        human_vel=0.08,
        turn_angle_deg=6.0,
        max_steps=500,
        world_size=10.0,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(log_dir: str, eval_env: gym.Env, disable_eval: bool):
    """
    Returns a list of SB3 callbacks:
        EvalCallback      — periodically tests the policy on a separate eval env
                            and saves the best model found so far.
        CheckpointCallback — saves a checkpoint every 10 000 steps so training
                             can be resumed if interrupted.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        CheckpointCallback(
            save_freq=10_000,
            save_path=CHECKPOINT_DIR,
            name_prefix="a2c_checkpoint",
            verbose=1,
        )
    ]

    if not disable_eval:
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=5,          # average over 5 episodes per evaluation
                eval_freq=5_000,            # evaluate every 5 000 steps
                best_model_save_path=os.path.join(log_dir, "best_model"),
                log_path=log_dir,
                deterministic=True,
                verbose=1,
            )
        )

    return callbacks


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """
    Instantiates the vectorised environment, builds the A2C model,
    attaches callbacks, runs training, and saves the final policy.
    """

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

    # --- 1. Vectorised training environments ---
    # make_vec_env spawns N_ENVS copies of make_env() and runs them in parallel.
    # This is safe because Environment has no shared mutable state.
    print(f"[train_a2c] Creating {args.n_envs} parallel training environments...")
    vec_train_env = make_vec_env(make_env, n_envs=args.n_envs)

    # A single deterministic environment for evaluation (not vectorised)
    eval_env = make_env()

    # --- 2. A2C model ---
    # "MlpPolicy" automatically creates:
    #   Actor  : Linear(4, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 16)  (logits over 16 actions)
    #   Critic : Linear(4, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 1)   (value estimate V(s))
    print("[train_a2c] Initialising A2C model...")
    model = A2C(
        policy=POLICY,
        env=vec_train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        tensorboard_log=args.log_dir,
        verbose=1,             # Print training stats to stdout
        device="auto",         # Use GPU if available, else CPU
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,  # Paper (Sec III-C) specifies ReLU, not Tanh
        ),
    )

    print(f"[train_a2c] Policy architecture:\n{model.policy}\n")

    # --- 3. Callbacks ---
    callbacks = build_callbacks(args.log_dir, eval_env, args.no_eval)

    # --- 4. Training loop ---
    print(f"[train_a2c] Starting training for {args.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="A2C",     # TensorBoard run folder name prefix
        reset_num_timesteps=True,
        progress_bar=True,     # Requires tqdm; shows a live progress bar
    )

    # --- 5. Save final model ---
    model.save(FINAL_MODEL_PATH)
    print(f"[train_a2c] Training complete. Model saved to → {FINAL_MODEL_PATH}.zip")
    print(f"[train_a2c] Visualise training with:\n  tensorboard --logdir {args.log_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
