import random
from simple_grid import transition, render, HUMAN_ACTIONS
from human_model import human_probabilities
from mcts import mcts

def run(steps=10, n_simulations=200):
    state = {
        'robot_pos': (3, 5),
        'human_pos': (3, 3),
        'human_heading': 'N'
    }

    render(state)

    for step in range(steps):
        robot_action = mcts(state, n_simulations=n_simulations)

        probs = human_probabilities(state)
        actions = list(probs.keys())
        weights = list(probs.values())
        human_action = random.choices(actions, weights=weights)[0]

        state = transition(state, robot_action, human_action)

        print(f"Step {step+1} | Robot: {robot_action} | Human: {human_action}")
        render(state)

if __name__ == '__main__':
    run()
