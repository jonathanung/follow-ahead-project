import os
import sys
import math
import numpy as np

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', 'RL_sim')))

import planner
from planner import MCTSPlanner, is_safe
from state import FollowState

def test_constants_alignment():
    print("Checking kinematic constants...")
    assert planner.ROBOT_VEL == 0.6
    assert planner.HUMAN_VEL == 0.6
    assert math.isclose(planner.ROBOT_TURN, math.radians(45.0))
    assert math.isclose(planner.HUMAN_TURN, math.radians(10.0))
    assert planner.SAFETY_R == 0.5
    print("PASS: test_constants_alignment")

def test_safety_logic_math():
    print("Checking Safety Logic (is_safe)...")
    # Human at origin, heading North
    # Robot at (0, 0.4) - directly on human, unsafe
    s_unsafe = FollowState(0,0,math.pi/2, 0, 0.4, math.pi/2)
    assert is_safe(s_unsafe) == False
    
    # Robot at (0, 1.0) - safe
    s_safe = FollowState(0,0,math.pi/2, 0, 1.5, math.pi/2)
    assert is_safe(s_safe) == True
    
    # Robot at (0, -0.4) - behind human. 
    # alpha = angle(human_heading, human_to_robot) 
    # human_heading = North (90 deg)
    # human_to_robot = South (-90 deg)
    # alpha = 180 deg
    # d_circle = a*cos(180) + sqrt(r^2 - a^2*sin^2(180)) = -0.25 + 0.5 = 0.25m
    # distance = 0.4m > 0.25m -> Safe
    s_behind = FollowState(0,0,math.pi/2, 0, -0.4, math.pi/2)
    assert is_safe(s_behind) == True
    print("PASS: test_safety_logic_math")

def test_mcts_pruning():
    print("Checking MCTS Safety Pruning...")
    # Human at (0,0) North
    # Robot at (0, -0.3) North
    # Action 'straight' moves 0.6m -> robot ends at (0, 0.3), which is d=0.3 < d_circle
    state = FollowState(0,0,math.pi/2, 0, -0.3, math.pi/2)
    
    planner_obj = MCTSPlanner(n_simulations=100, verbose=True)
    action = planner_obj.plan(state)
    
    print(f"Action chosen from dangerous start: {action}")
    # 'straight' should definitely NOT be the chosen action if pruned correctly
    assert action != 'straight'
    assert action != 'fast_straight'
    print("PASS: test_mcts_pruning")

if __name__ == '__main__':
    try:
        test_constants_alignment()
        test_safety_logic_math()
        test_mcts_pruning()
        print("\nALL ALIGNMENT TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
