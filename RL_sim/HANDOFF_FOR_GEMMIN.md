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

## Setup

```bash
pip install stable-baselines3 torch
```

---

## Usage in mcts_node.py

```python
import sys
sys.path.insert(0, '<path-to-repo>/RL_sim')
from RL_interface import RL_model
import numpy as np

# Load once at startup
rl = RL_model()
rl.load_model('<path-to-repo>/RL_sim/models/a2c_follow_ahead')

# Inside evaluate_node() — build obs from MCTS state
state_arr = node.state.state   # shape (2,3): [[xr,yr,θr], [xh,yh,θh]]
obs = np.array([
    state_arr[0,0] - state_arr[1,0],  # dx  (robot_x - human_x)
    state_arr[0,1] - state_arr[1,1],  # dy  (robot_y - human_y)
    state_arr[1,2],                   # human_theta (rad)
    state_arr[0,2],                   # robot_theta (rad)
], dtype=np.float32)

value = rl.evaluate_state(obs)   # → float scalar V(s)
return r + value / 10.0 * params['gamma']
```

---

## Quick test

```bash
cd RL_sim/
python -c "
from RL_interface import RL_model
import numpy as np
rl = RL_model()
rl.load_model('models/a2c_follow_ahead')
v = rl.evaluate_state(np.zeros(4, dtype=np.float32))
print('V(zeros):', v)   # should print a float
"
```

---

Questions → Ankush
