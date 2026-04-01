import os
import sys
import math
import numpy as np
# import pytest

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', 'RL_sim')))

from planner import MCTSPlanner, ROBOT_ACTIONS
from state import FollowState

def test_mcts_initial_plan():
    """Verify that MCTS can run from a cold start and pick a valid action."""
    planner = MCTSPlanner(n_simulations=100, verbose=True)
    
    # 1.5m behind human, heading North
    state = FollowState(
        human_x=0.0, human_y=0.0, human_theta=math.pi/2,
        robot_x=0.0, robot_y=-1.5, robot_theta=math.pi/2
    )
    
    action = planner.plan(state)
    print(f"Test Initial Plan: {action}")
    assert action in ROBOT_ACTIONS or action == 'STAY'

def test_mcts_straight_vicon():
    """Simulate a perfect straight walk and see if the planner stays straight."""
    planner = MCTSPlanner(n_simulations=500, verbose=False)
    
    # Ideal position: 1.5m ahead of human (North world frame)
    # Human heading North (pi/2)
    # Robot heading North (pi/2) at x=0, y=1.5
    state = FollowState(
        human_x=0.0, human_y=0.0, human_theta=math.pi/2,
        robot_x=0.0, robot_y=1.5, robot_theta=math.pi/2
    )
    
    # With perfect positioning, the best action should be 'straight' or 'fast_straight' 
    # to maintain the 1.5m gap.
    action = planner.plan(state)
    print(f"Test Straight Walk Action: {action}")
    assert action in ['straight', 'fast_straight']

def test_mcts_sharp_turn():
    """Verify that MCTS adapts to a human turn."""
    planner = MCTSPlanner(n_simulations=1000, verbose=False)
    
    # Human is turning right (from North to East)
    # The robot was ahead (y=1.5, x=0), but hasn't turned yet.
    state = FollowState(
        human_x=0.0, human_y=0.0, human_theta=0.0, # Human turned East
        robot_x=0.0, robot_y=1.5, robot_theta=math.pi/2 # Robot still North
    )
    
    # LSTM prior: human is likely turning right
    human_probs = {'right': 0.9, 'straight': 0.05, 'left': 0.05}
    
    action = planner.plan(state, human_probs=human_probs)
    print(f"Test Sharp Turn Action: {action}")
    # The robot should turn right to track the human's new heading
    assert 'right' in action

if __name__ == '__main__':
    # Run tests manually
    try:
        test_mcts_initial_plan()
        print("PASS: test_mcts_initial_plan")
        test_mcts_straight_vicon()
        print("PASS: test_mcts_straight_vicon")
        test_mcts_sharp_turn()
        print("PASS: test_mcts_sharp_turn")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)
