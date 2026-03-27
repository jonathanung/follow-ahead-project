"""
test_integration.py — Integration smoke test for RL value function + MCTS planner.

Verifies:
  1. RL_interface loads the trained model without error
  2. evaluate_state() returns a float for a known-good and known-bad pose
  3. V(good pose) > V(bad pose)  — value function discriminates correctly
  4. paper_reward() returns correct values at known poses
  5. MCTSPlanner.plan() returns a valid action in one planning call

Run from follow/scripts/:
    cd follow/scripts/
    python test_integration.py

Or from the repo root:
    python follow/scripts/test_integration.py
"""

import sys
import os
import math
import numpy as np

# Ensure follow/scripts and RL_sim are on the path
_HERE      = os.path.dirname(os.path.abspath(__file__))
_RL_SIM    = os.path.abspath(os.path.join(_HERE, '..', '..', 'RL_sim'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _RL_SIM)

from RL_interface import RL_model
from planner import MCTSPlanner, _paper_reward, rl_value, ROBOT_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(_RL_SIM, 'models', 'a2c_follow_ahead')

# Good state: robot is directly ahead of human at 1.5m (ideal)
#   human at grid cell (3,3) heading 'N' → robot should be at (3,4)
GOOD_STATE = {
    'robot_pos':     (3, 4),   # one cell north of human → directly ahead
    'human_pos':     (3, 3),
    'human_heading': 'N',
}

# Bad state: robot is directly behind human at 4m (worst)
#   human at (3,3) heading 'N' → robot at (3,-5) is far behind
BAD_STATE = {
    'robot_pos':     (3, 0),   # south of human → behind
    'human_pos':     (3, 3),
    'human_heading': 'N',
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rl_model_loads():
    rl = RL_model()
    rl.load_model(MODEL_PATH)
    assert rl.model is not None, "Model did not load"
    print("  [PASS] RL model loaded")
    return rl


def test_evaluate_state(rl):
    # Good pose: robot directly ahead of human at ~0.5m (1 cell * 0.5m)
    # obs = [dx, dy, h_theta, r_theta]
    good_obs = np.array([0.0, 0.5, math.pi / 2, 0.0], dtype=np.float32)   # directly ahead 0.5m N
    bad_obs  = np.array([0.0, -1.5, math.pi / 2, 0.0], dtype=np.float32)  # behind at -1.5m

    v_good = rl.evaluate_state(good_obs)
    v_bad  = rl.evaluate_state(bad_obs)

    assert isinstance(v_good, float), f"Expected float, got {type(v_good)}"
    assert isinstance(v_bad,  float), f"Expected float, got {type(v_bad)}"
    print(f"  [INFO] V(good obs) = {v_good:.4f},  V(bad obs) = {v_bad:.4f}")

    # Value function should score 'good' higher than 'bad'
    assert v_good > v_bad, (
        f"Value function should score ideal pose higher than bad pose, "
        f"but V(good)={v_good:.4f} <= V(bad)={v_bad:.4f}"
    )
    print(f"  [PASS] V(good) > V(bad)  ({v_good:.4f} > {v_bad:.4f})")


def test_rl_value_with_grid_state():
    v_good = rl_value(GOOD_STATE)
    v_bad  = rl_value(BAD_STATE)
    assert isinstance(v_good, float)
    assert isinstance(v_bad,  float)
    print(f"  [PASS] rl_value(grid state): V(good)={v_good:.4f}  V(bad)={v_bad:.4f}")


def test_paper_reward():
    r_good = _paper_reward(GOOD_STATE)
    r_bad  = _paper_reward(BAD_STATE)
    assert r_good > r_bad, (
        f"Paper reward should be higher for good state: "
        f"r(good)={r_good:.4f}, r(bad)={r_bad:.4f}"
    )
    print(f"  [PASS] paper_reward: r(good)={r_good:.4f} > r(bad)={r_bad:.4f}")


def test_planner_returns_valid_action():
    planner = MCTSPlanner(n_simulations=200, verbose=False)
    state = {
        'robot_pos':     (3, 2),
        'human_pos':     (3, 3),
        'human_heading': 'N',
    }
    action = planner.plan(state)
    valid_actions = list(ROBOT_ACTIONS) + ['STAY']
    assert action in valid_actions, f"Invalid action: {action!r}"
    print(f"  [PASS] MCTSPlanner.plan() returned valid action: '{action}'")


def test_planner_with_mock_lstm():
    planner = MCTSPlanner(n_simulations=200, verbose=False)
    state = {
        'robot_pos':     (3, 2),
        'human_pos':     (3, 3),
        'human_heading': 'N',
    }
    # Mock LSTM output strongly predicting a right turn
    mock_probs = {'left': 0.05, 'straight': 0.05, 'right': 0.90}
    action = planner.plan(state, human_probs=mock_probs)
    valid_actions = list(ROBOT_ACTIONS) + ['STAY']
    assert action in valid_actions, f"Invalid action: {action!r}"
    print(f"  [PASS] MCTSPlanner.plan(human_probs=...) returned: '{action}'")


def test_real_lstm_inference():
    # Attempt to load the real LSTM model and run one prediction
    try:
        from lstm_fc import HumanActionPredictor, TrajectoryBuffer
    except ImportError:
        print("  [SKIP] lstm_fc package not in PYTHONPATH")
        return

    model_path = os.path.join(_HERE, '..', '..', 'lstm-fc', 'outputs', 'final_v3', 'model_final.pt')
    if not os.path.exists(model_path):
        print(f"  [SKIP] LSTM model not found at {model_path}")
        return

    predictor = HumanActionPredictor(model_path, device="cpu")
    print("  [PASS] Loaded HumanActionPredictor")
    
    # 14 points of someone walking straight North
    history = [[3.0, float(i)] for i in range(14)]
    probs = predictor.predict(history)
    
    assert 'straight' in probs and 'left' in probs and 'right' in probs
    print(f"  [PASS] Real LSTM inference output: {probs}")
    
    # Pass to planner
    planner = MCTSPlanner(n_simulations=50, verbose=False)
    state = {'robot_pos': (3, 2), 'human_pos': (3, 3), 'human_heading': 'N'}
    action = planner.plan(state, human_probs=probs)
    print(f"  [PASS] MCTSPlanner using real LSTM probs returned: '{action}'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== RL-MCTS Integration Smoke Tests ===\n")
    tests = [
        ("RL model loads",                 lambda: test_rl_model_loads()),
        ("evaluate_state() returns float", lambda: test_evaluate_state(test_rl_model_loads())),
        ("rl_value() with grid state",     lambda: test_rl_value_with_grid_state()),
        ("paper_reward() discriminates",   lambda: test_paper_reward()),
        ("MCTSPlanner.plan() valid action", lambda: test_planner_returns_valid_action()),
        ("MCTSPlanner.plan() with mock LSTM", lambda: test_planner_with_mock_lstm()),
        ("Real LSTM inference + MCTS logic",  lambda: test_real_lstm_inference()),
    ]

    passed = 0
    for name, fn in tests:
        print(f"[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")

    print(f"\n{'='*42}")
    print(f"Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("All tests passed ✓")
    else:
        sys.exit(1)
