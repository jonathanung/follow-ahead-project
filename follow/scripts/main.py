import random
from simple_grid import transition, render, desired_robot_pos
from human_model import human_probabilities
from planner import MCTSPlanner

def run(steps=50, n_simulations=None, time_budget=None, verbose=False):
    state = {
        'robot_pos': (3, 4),
        'human_pos': (3, 3),
        'human_heading': 'N',
    }
    planner = MCTSPlanner(n_simulations=n_simulations, time_budget=time_budget, verbose=verbose)
    render(state, step=0)

    for step in range(1, steps + 1):
        robot_action = planner.plan(state)
        probs = human_probabilities(state)
        human_action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        state = transition(state, robot_action, human_action)
        render(state, step=step, robot_action=robot_action, human_action=human_action)

if __name__ == '__main__':
    run(time_budget=0.15)
