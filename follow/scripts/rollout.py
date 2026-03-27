import random
from simple_grid import transition, reward, desired_robot_pos, move_in_dir, ROBOT_ACTIONS
from human_model import human_probabilities

def _greedy_robot_action(state):
    desired = desired_robot_pos(state['human_pos'], state['human_heading'])
    rp = state['robot_pos']
    best_action, best_dist = None, float('inf')
    for action in ROBOT_ACTIONS:
        pos = move_in_dir(rp, action)
        dist = ((pos[0] - desired[0])**2 + (pos[1] - desired[1])**2)**0.5
        if dist < best_dist:
            best_dist, best_action = dist, action
    return best_action

def rollout(state, depth=10):
    total = 0.0
    s = state
    for _ in range(depth):
        probs = human_probabilities(s)
        h_action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        s = transition(s, _greedy_robot_action(s), h_action)
        total += reward(s)
    return total
