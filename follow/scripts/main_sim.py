from simple_grid import transition, render
from planner import MCTSPlanner

SCENARIOS = {
    'straight': ['forward'] * 10,
    'turn_left': ['forward', 'forward', 'left', 'forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'forward'],
    'zigzag':   ['forward', 'left', 'forward', 'right', 'forward', 'left', 'forward', 'right', 'forward', 'forward'],
    'u_turn':   ['forward', 'forward', 'left', 'left', 'forward', 'forward', 'forward', 'forward', 'forward', 'forward'],
}

def run_scenario(name, human_actions, n_simulations=1000, verbose=False):
    state = {
        'robot_pos': (3, 4),
        'human_pos': (3, 3),
        'human_heading': 'N',
    }
    planner = MCTSPlanner(n_simulations=n_simulations, verbose=verbose)
    print(f"=== {name} ===")
    render(state, step=0)

    for step, human_action in enumerate(human_actions, 1):
        robot_action = planner.plan(state)
        state = transition(state, robot_action, human_action)
        render(state, step=step, robot_action=robot_action, human_action=human_action)

if __name__ == '__main__':
    for name, actions in SCENARIOS.items():
        run_scenario(name, actions)
