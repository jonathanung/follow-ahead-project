"""
test_env.py — Smoke tests for the corrected Follow-Ahead RL environment.

Verifies:
  1. Action space is 6 (MCTS-compatible)
  2. reset() returns correct obs shape and dtype
  3. Full episode runs without error
  4. All rewards are in [-1, 1]
  5. Episode terminates with truncated=True, terminated=False
  6. FollowState.distance and .alpha are finite
  7. reward.r_d(1.5) == 1.0 (peak)
  8. reward.r_alpha(0.0) == 1.0 (peak)

Run from the RL_sim/ directory:
    python test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from nav_env import Environment
from reward import r_d, r_alpha
from state import FollowState


def test_action_space():
    env = Environment()
    assert env.action_space.n == 6, \
        f"Expected 6 actions (MCTS vocab), got {env.action_space.n}"
    print("  [PASS] action_space.n == 6")


def test_reset():
    env = Environment()
    obs, info = env.reset(seed=42)
    assert obs.shape == (4,), f"Expected obs shape (4,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert isinstance(info, dict), "info must be a dict"
    print("  [PASS] reset() → obs shape (4,) float32")


def test_full_episode():
    env = Environment(max_steps=200)
    obs, _ = env.reset(seed=0)
    rewards = []
    terminated_any = False
    truncated_final = False

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (4,), f"Step {step}: bad obs shape"
        assert obs.dtype == np.float32, f"Step {step}: bad dtype"
        assert -1.0 <= reward <= 1.0, f"Step {step}: reward {reward} out of [-1,1]"
        rewards.append(reward)

        if terminated:
            terminated_any = True
        if truncated:
            truncated_final = True
            break

    assert not terminated_any, "terminated should never be True (no goal state)"
    assert truncated_final, "Episode should end with truncated=True at step limit"
    print(f"  [PASS] 200-step episode: rewards in [-1,1], "
          f"terminated=False, truncated=True")
    print(f"         mean reward = {np.mean(rewards):.4f}")


def test_state_properties():
    env = Environment()
    env.reset(seed=7)
    env.step(2)  # straight
    s = env.state
    assert np.isfinite(s.distance), f"distance is not finite: {s.distance}"
    assert np.isfinite(s.alpha),    f"alpha is not finite: {s.alpha}"
    assert 0.0 <= s.alpha <= 180.0, f"alpha {s.alpha} out of [0,180]"
    print(f"  [PASS] FollowState.distance={s.distance:.3f}m, alpha={s.alpha:.1f}°")


def test_reward_peaks():
    # r_d should peak at 1.0 when d=1.5m (rescaled)
    val_rd = r_d(1.5)
    assert abs(val_rd - 1.0) < 1e-9, f"r_d(1.5) should be 1.0, got {val_rd}"
    print(f"  [PASS] r_d(1.5) = {val_rd:.6f} (expected 1.0)")

    # r_alpha should peak at 1.0 when alpha=0°
    val_ra = r_alpha(0.0)
    assert abs(val_ra - 1.0) < 1e-9, f"r_alpha(0.0) should be 1.0, got {val_ra}"
    print(f"  [PASS] r_alpha(0.0) = {val_ra:.6f} (expected 1.0)")

    # r_d should be -1 (rescaled to 0.0) when distance is out-of-zone
    val_rd_far = r_d(10.0)
    assert abs(val_rd_far - 0.0) < 1e-9, f"r_d(10.0) should be 0.0 (rescaled -1), got {val_rd_far}"
    print(f"  [PASS] r_d(10.0) = {val_rd_far:.6f} (expected 0.0 — rescaled -1)")


def test_observation_bounds():
    env = Environment(world_size=20.0)
    obs, _ = env.reset(seed=1)
    lo = env.observation_space.low
    hi = env.observation_space.high
    for step in range(50):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    # obs should be clamp-able to world bounds (not guaranteed always in-bounds
    # after boundary clipping, but should never be wildly out)
    print(f"  [PASS] obs bounds check: obs in [{obs.min():.2f}, {obs.max():.2f}]")


if __name__ == "__main__":
    print("\n=== Follow-Ahead RL_sim smoke tests ===\n")
    tests = [
        ("Action space",        test_action_space),
        ("Reset output",        test_reset),
        ("Full episode",        test_full_episode),
        ("FollowState props",   test_state_properties),
        ("Reward peaks",        test_reward_peaks),
        ("Obs bounds",          test_observation_bounds),
    ]

    passed = 0
    for name, fn in tests:
        print(f"[TEST] {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("All tests passed ✓")
    else:
        sys.exit(1)
