import random
from simple_grid import transition, reward, desired_robot_pos, move_in_dir, turn_left, turn_right, ROBOT_ACTIONS, HUMAN_ACTIONS
from human_model import human_probabilities

def _greedy_robot_action(state):
    desired = desired_robot_pos(state['human_pos'], state['human_heading'])
    best_action = None
    best_dist = float('inf')
    hh = state['human_heading']
    rp = state['robot_pos']

    candidates = {
        'forward': move_in_dir(rp, hh),
        'left':    move_in_dir(rp, turn_left(hh)),
        'right':   move_in_dir(rp, turn_right(hh)),
    }

    for action, pos in candidates.items():
        dist = ((pos[0] - desired[0])**2 + (pos[1] - desired[1])**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_action = action

    return best_action

def rollout(state, depth=10):
    total = 0.0
    s = state
    for _ in range(depth):
        probs = human_probabilities(s)
        h_action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        r_action = _greedy_robot_action(s)
        s = transition(s, r_action, h_action)
        total += reward(s)
    return total
