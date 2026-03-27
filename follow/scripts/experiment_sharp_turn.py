"""
experiment_sharp_turn.py — Demonstrates MCTS+RL+LSTM integration

This script runs a full simulation loop in the mathematical grid world (simple_grid.py)
without needing ROS or Gazebo. It proves the algorithmic core works:

1. A simulated human walks forward, then makes a sharp right turn.
2. The TrajectoryBuffer records it and HumanActionPredictor (LSTM) predicts future actions.
3. The MCTSPlanner uses the LSTM prior + RL value to plan the robot's action.
4. The terminal visuals plot the robot successfully adapting to the sharp turn.
"""

import sys
import os
import time
import numpy as np

# Ensure follow/scripts, RL_sim, and lstm-fc are on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', 'RL_sim')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', 'lstm-fc')))

from planner import MCTSPlanner, _paper_reward
from simple_grid import transition, render
from lstm_fc.inference import HumanActionPredictor, TrajectoryBuffer
from lstm_fc.actions import INPUT_LENGTH

def run_experiment():
    print("=== INITIALIZING MODELS ===")
    
    # 1. Initialize MCTS Planner (RL model loads automatically inside planner.py)
    planner = MCTSPlanner(n_simulations=1000, verbose=False)
    
    # 2. Initialize LSTM Predictor
    model_path = os.path.join(_HERE, '..', '..', 'lstm-fc', 'outputs', 'final_v3', 'model_final.pt')
    if not os.path.exists(model_path):
        print(f"[ERROR] LSTM model not found at {model_path}. Cannot run experiment.")
        return
    lst_predictor = HumanActionPredictor(model_path, device="cpu")
    traj_buf = TrajectoryBuffer(length=INPUT_LENGTH)
    
    print("\n=== STARTING EXPERIMENT = Sharp Turn ===\n")
    
    # Initial State: Human at bottom, heading North. Robot slightly ahead.
    state = {
        'robot_pos': (3, 2),
        'human_pos': (3, 1),
        'human_heading': 'N'
    }
    
    # Render initial
    render(state, step=0)
    
    total_reward = 0.0
    
    # Pre-fill trajectory buffer so LSTM works instantly (simulate human walking from behind)
    for y in range(INPUT_LENGTH):
        traj_buf.push(3.0, 1.0 - (INPUT_LENGTH - y) * 0.5)
    
    # Simulate a sharp turn scenario
    # Human moves: 3x forward, 2x left turn, 3x forward
    human_script = ['forward', 'forward', 'forward', 'right', 'right', 'forward', 'forward']
    
    for step, h_action in enumerate(human_script, start=1):
        # 1. Update Human history for LSTM
        hp = state['human_pos']
        traj_buf.push(float(hp[0]), float(hp[1]))
        
        # 2. LSTM Prediction
        if traj_buf.ready:
            human_probs = lst_predictor.predict(traj_buf.get())
        else:
            human_probs = None
            
        # 3. MCTS Decision (using RL value + LSTM priors)
        r_action = planner.plan(state, human_probs=human_probs)
        
        # 4. Environment Transition
        state = transition(state, r_action, h_action)
        
        # Log reward
        r = _paper_reward(state)
        total_reward += r
        
        # 5. Verify & Render
        render(state, step=step, robot_action=r_action, human_action=h_action)
        if human_probs:
            print(f"LSTM Prediction: {human_probs}")
        time.sleep(0.5)  # brief pause for terminal observation
        
    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Total Paper Reward Accumulated: {total_reward:.4f}")
    
if __name__ == '__main__':
    run_experiment()
