#Purpose: turns global coordinates to relative coordinates for human and robot, normalizes them too

import numpy as np

def state_to_obs(s, next_to_move):
    """
    Transforms raw global coordinates into relative observations for the RL/LSTM models.
    s structure:
    s[0]: [robot_x, robot_y, robot_theta]
    s[1]: [human_x, human_y, human_theta]
    s[2]: [target_x, target_y, target_theta] (Look-ahead/Human next pos)
    """
    # [UPDATE] next_to_move == 1 indicates processing for the robot's logic
    if next_to_move == 1:   
        # gamma: Angle of the vector from Human (s[1]) to Target (s[2]), 
        # then subtract Human's current global orientation (s[1,2]).
        # This represents the human's relative heading.
        gamma = np.arctan2(s[2][1]-s[1][1], s[2][0]-s[1][0]) - s[1][2]
        
        # dis_to_rob: Euclidean distance between human and robot
        # Uses only the first two columns (x, y).
        dis_to_rob = np.linalg.norm(s[1,:2] - s[0, :2])
        
        # angle_to_robot: Relative bearing
        # Global angle of the vector from Human (s[1]) to Robot (s[0]).
        angle_to_robot = np.arctan2(s[0,1]-s[1,1], s[0,0]-s[1,0])
        
        # angle: Human's orientation (s[1,2]) relative to the line-of-sight to the robot.
        angle = s[1,2] - angle_to_robot
        # beta: Robot's orientation (s[0,2]) relative to the line-of-sight to the human.
        beta = s[0,2] - angle_to_robot
        
        # [UPDATE] Explicitly ensuring float32 return for PyTorch compatibility
        return np.array([gamma, dis_to_rob, angle, beta], dtype=np.float32)
    
    else:
        # deltaP: (x, y) displacement between Human (s[1]) and Robot (s[0]).
        deltaP = s[1, :] - s[0, :]
        # gamma: Change in human's heading relative to their current orientation.
        # arctan2(target_y - human_y, target_x - human_x) - human_theta.
        gamma = np.arctan2((s[2,1]-s[1,1]), (s[2,0]-s[1,0])) - s[1,2]
        
        # [UPDATE] Modernized array concatenation
        return np.append(deltaP, gamma).astype(np.float32)

def normalize(data, mean_std, next_to_move):
    """
    Scales observations to have zero mean and unit variance.
    """
    if next_to_move == 1:
        mean = mean_std['human_mean']
        std = mean_std['human_std']
    else:
        mean = mean_std['robot_mean']
        std = mean_std['robot_std']

    # [UPDATE] Added a small epsilon (1e-8) to avoid division by zero errors
    return (data - mean) / (std + 1e-8)