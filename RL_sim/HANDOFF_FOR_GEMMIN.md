# RL Value Function — Handoff for Gemmin

**From**: Ankush Singh (RL lead)  
**For**: Gemmin Sugiura (MCTS / ROS 2 C++ node)

---

## What this module provides

A trained A2C value function `V(s)` that scores how good a robot–human configuration is, for use during MCTS node expansion and backpropagation.

---

## Files to use

| File | Description |
|---|---|
| `models/value_fn.pt` | **TorchScript** — recommended for C++ libtorch |
| `models/value_fn.onnx` | **ONNX** — alternative, needs onnxruntime |
| `models/a2c_follow_ahead.zip` | Full SB3 model (Python only, do not use in C++) |
| `RL_interface.py` | Python wrapper (for Python MCTS / testing only) |

---

## Input / Output spec

| | Shape | Dtype | Description |
|---|---|---|---|
| **Input** | `[1, 4]` | `float32` | `[dx, dy, human_theta, robot_theta]` |
| **Output** | `[1, 1]` | `float32` | Scalar value estimate V(s) |

### Observation meaning
```
dx           = robot_x - human_x        (metres)
dy           = robot_y - human_y        (metres)
human_theta  = human global heading     (radians)
robot_theta  = robot global heading     (radians)
```

---

## C++ Integration (libtorch / TorchScript)

```cpp
#include <torch/script.h>

// Load once at startup
torch::jit::script::Module value_fn = torch::jit::load("value_fn.pt");
value_fn.eval();

// Call inside evaluate_node()
auto obs = torch::tensor({dx, dy, human_theta, robot_theta},
                          torch::kFloat32).unsqueeze(0); // shape [1,4]
auto output = value_fn.forward({obs}).toTensor();        // shape [1,1]
float v = output[0][0].item<float>();                    // scalar V(s)
```

### CMakeLists.txt addition
```cmake
find_package(Torch REQUIRED)
target_link_libraries(mcts_node ${TORCH_LIBRARIES})
```

---

## Python testing (confirm it works before C++ integration)

```bash
cd RL_sim/
python - <<'EOF'
import torch
m = torch.jit.load("models/value_fn.pt")
m.eval()

# Ideal pose: robot directly ahead of human at ~1.5m
import math
ideal = torch.tensor([[0.0, 1.5, 0.0, 0.0]])     # ahead at 1.5m
random = torch.tensor([[5.0, -3.0, 0.5, -0.2]])  # bad pose

print(f"V(ideal)  = {m(ideal)[0,0].item():.4f}")
print(f"V(random) = {m(random)[0,0].item():.4f}")
# V(ideal) should be HIGHER than V(random)
EOF
```

---

## How it fits into MCTS (search.py)

In `evaluate_node()`, replace the existing RL call with:

```python
# Build 4-dim obs from MCTS node state
state_arr = node.state.state            # shape (2, 3): [[xr,yr,θr], [xh,yh,θh]]
obs = np.array([
    state_arr[0, 0] - state_arr[1, 0], # dx
    state_arr[0, 1] - state_arr[1, 1], # dy
    state_arr[1, 2],                   # human_theta
    state_arr[0, 2],                   # robot_theta
], dtype=np.float32)

value = self.params['RL_model'].evaluate_state(obs)  # → float scalar

return r + value / 10.0 * self.params['gamma']
```

---

## Regenerating the exported models

If you need to retrain or re-export:

```bash
cd RL_sim/
python train_a2c.py               # re-trains (500k steps, ~60s)
python export_value_fn.py        # re-exports .pt and .onnx
```

---

## Contact

Any questions about the obs format or reward shaping — reach out to Ankush.
