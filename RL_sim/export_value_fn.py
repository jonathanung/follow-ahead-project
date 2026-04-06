"""
export_value_fn.py — Export the trained A2C value function for Gemmin / ROS 2 C++
====================================================================================

The project proposal specifies that the RL value function must be delivered as a
TorchScript (.pt) or ONNX (.onnx) file so Gemmin can load it in the C++ MCTS node
without a Python / Stable-Baselines3 dependency.

This script does three things:
  1. Loads the trained SB3 A2C model (a2c_follow_ahead.zip).
  2. Extracts just the CRITIC network (value function V(s)).
  3. Exports it as:
       - TorchScript  →  models/value_fn.pt      (preferred for C++ libtorch)
       - ONNX         →  models/value_fn.onnx    (alternative, broader tooling)

Usage
-----
    cd RL_sim/
    python export_value_fn.py

Both output files will be placed in RL_sim/models/.

C++ loading example (TorchScript)
----------------------------------
    torch::jit::script::Module model = torch::jit::load("value_fn.pt");
    model.eval();
    auto input = torch::tensor({dx, dy, human_theta, robot_theta}).unsqueeze(0);
    auto value = model.forward({input}).toTensor();   // shape [1,1]
    float v = value[0][0].item<float>();

ONNX loading example (onnxruntime)
------------------------------------
    #include <onnxruntime_cxx_api.h>
    // session.Run(...) with input shape [1, 4] float32
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "models", "a2c_follow_ahead")
OUT_DIR      = os.path.join(SCRIPT_DIR, "models")
TS_OUT       = os.path.join(OUT_DIR, "value_fn.pt")
ONNX_OUT     = os.path.join(OUT_DIR, "value_fn.onnx")

# ─── Input spec ─────────────────────────────────────────────────────────────
# The critic takes a (1, 4) float32 tensor:
#   [dx, dy, human_theta, robot_theta]
OBS_DIM = 4


# ─── Thin wrapper so TorchScript traces only the critic ──────────────────────
class CriticWrapper(nn.Module):
    """
    Wraps the SB3 MLP critic so torch.jit.trace / torch.onnx.export
    sees a single forward(obs) → scalar value call.
    """
    def __init__(self, policy):
        super().__init__()
        # SB3 MlpPolicy stores the MLP feature extractor + value head separately
        self.mlp_extractor   = policy.mlp_extractor
        self.value_net       = policy.value_net
        self.features_extractor = policy.features_extractor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: shape (1, 4) float32 → returns shape (1, 1) float32"""
        features = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        value = self.value_net(latent_vf)
        return value


def load_sb3_model(model_path: str) -> A2C:
    print(f"[export] Loading SB3 A2C from: {model_path}.zip")
    model = A2C.load(model_path)
    model.policy.set_training_mode(False)
    return model


def export_torchscript(critic: CriticWrapper, out_path: str) -> None:
    dummy_input = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    # Use trace (not script) — SB3 networks aren't fully scriptable
    traced = torch.jit.trace(critic, dummy_input)
    traced.save(out_path)
    print(f"[export] TorchScript saved → {out_path}")

    # Sanity check
    loaded = torch.jit.load(out_path)
    loaded.eval()
    v = loaded(dummy_input)
    assert v.shape == (1, 1), f"Unexpected output shape: {v.shape}"
    print(f"[export] TorchScript sanity check: V(zeros) = {v.item():.4f} ✓")


def export_onnx(critic: CriticWrapper, out_path: str) -> None:
    dummy_input = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        critic,
        dummy_input,
        out_path,
        input_names=["obs"],
        output_names=["value"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "value": {0: "batch"}},
    )
    print(f"[export] ONNX saved → {out_path}")

    # Sanity check with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path)
        dummy_np = np.zeros((1, OBS_DIM), dtype=np.float32)
        v = sess.run(["value"], {"obs": dummy_np})[0]
        assert v.shape == (1, 1), f"Unexpected ONNX output shape: {v.shape}"
        print(f"[export] ONNX sanity check (onnxruntime): V(zeros) = {v[0,0]:.4f} ✓")
    except ImportError:
        print("[export] onnxruntime not installed — skipping ONNX sanity check.")
        print("[export]   Install with:  pip install onnxruntime")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model  = load_sb3_model(MODEL_PATH)
    critic = CriticWrapper(model.policy)
    critic.eval()

    export_torchscript(critic, TS_OUT)
    export_onnx(critic, ONNX_OUT)

    print()
    print("=" * 60)
    print("Export complete. Gemmin can load either file:")
    print(f"  TorchScript  →  {TS_OUT}")
    print(f"  ONNX         →  {ONNX_OUT}")
    print()
    print("Input spec:  float32 tensor, shape [1, 4]")
    print("             [dx, dy, human_theta, robot_theta]")
    print("Output spec: float32 tensor, shape [1, 1]  — V(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
