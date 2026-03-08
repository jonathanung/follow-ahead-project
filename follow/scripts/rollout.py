import random
from simple_grid import transition, reward, ROBOT_ACTIONS, HUMAN_ACTIONS
from human_model import human_probabilities

def rollout(state, depth=5):
    total = 0.0
    s = state
    for _ in range(depth):
        probs = human_probabilities(s)
        actions = list(probs.keys())
        weights = list(probs.values())
        h_action = random.choices(actions, weights=weights)[0]
        r_action = random.choice(ROBOT_ACTIONS)
        s = transition(s, r_action, h_action)
        total += reward(s)
    return total
