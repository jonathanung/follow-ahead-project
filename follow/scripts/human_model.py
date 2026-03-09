from simple_grid import move_in_dir

def human_probabilities(state):
    hp = state['human_pos']
    hh = state['human_heading']
    forward_pos = move_in_dir(hp, hh)

    if forward_pos == hp:
        return {'forward': 0.1, 'left': 0.45, 'right': 0.45}

    return {'forward': 0.85, 'left': 0.075, 'right': 0.075}
