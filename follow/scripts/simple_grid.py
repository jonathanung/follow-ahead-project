GRID_SIZE = 7
DIRECTIONS = ['N', 'E', 'S', 'W']
DIR_VECTORS = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
ROBOT_ACTIONS = ['forward', 'left', 'right']
HUMAN_ACTIONS = ['forward', 'left', 'right']

def turn_left(heading):
    return DIRECTIONS[(DIRECTIONS.index(heading) - 1) % 4]

def turn_right(heading):
    return DIRECTIONS[(DIRECTIONS.index(heading) + 1) % 4]

def move_in_dir(pos, direction):
    dx, dy = DIR_VECTORS[direction]
    nx = max(0, min(GRID_SIZE - 1, pos[0] + dx))
    ny = max(0, min(GRID_SIZE - 1, pos[1] + dy))
    return (nx, ny)

def desired_robot_pos(human_pos, human_heading):
    return move_in_dir(human_pos, human_heading)

def transition(state, robot_action, human_action):
    rp = state['robot_pos']
    hp = state['human_pos']
    hh = state['human_heading']

    if human_action == 'forward':
        new_hp, new_hh = move_in_dir(hp, hh), hh
    elif human_action == 'left':
        new_hp, new_hh = hp, turn_left(hh)
    else:
        new_hp, new_hh = hp, turn_right(hh)

    if robot_action == 'forward':
        new_rp = move_in_dir(rp, new_hh)
    elif robot_action == 'left':
        new_rp = move_in_dir(rp, turn_left(new_hh))
    else:
        new_rp = move_in_dir(rp, turn_right(new_hh))

    if new_rp == new_hp:
        new_rp = rp

    return {'robot_pos': new_rp, 'human_pos': new_hp, 'human_heading': new_hh}

def reward(state):
    desired = desired_robot_pos(state['human_pos'], state['human_heading'])
    rp = state['robot_pos']
    dist = ((rp[0] - desired[0])**2 + (rp[1] - desired[1])**2)**0.5
    return -(dist ** 2)

def _heatmap_char(x, y, desired, human_pos, robot_pos):
    if (x, y) == desired:
        return '*'
    if (x, y) == human_pos:
        return 'H'
    if (x, y) == robot_pos:
        return 'R'
    dist = ((x - desired[0])**2 + (y - desired[1])**2)**0.5
    max_dist = ((GRID_SIZE - 1)**2 + (GRID_SIZE - 1)**2)**0.5
    intensity = int((1 - dist / max_dist) * 9)
    return str(intensity)

def render(state, step=None, robot_action=None, human_action=None):
    hp = state['human_pos']
    rp = state['robot_pos']
    hh = state['human_heading']
    desired = desired_robot_pos(hp, hh)
    r = reward(state)

    grid_rows = []
    heat_rows = []

    for y in range(GRID_SIZE):
        grow = []
        hrow = []
        for x in range(GRID_SIZE):
            if (x, y) == hp:
                grow.append('H')
            elif (x, y) == rp:
                grow.append('R')
            else:
                grow.append('.')
            hrow.append(_heatmap_char(x, y, desired, hp, rp))
        grid_rows.append(' '.join(grow))
        heat_rows.append(' '.join(hrow))

    if step is not None:
        header = f"step {step}"
        if robot_action and human_action:
            header += f"  |  robot: {robot_action:<8} human: {human_action}"
        print(header)

    sep = '     '
    for g, h in zip(grid_rows, heat_rows):
        print(g + sep + h)

    print(f"heading: {hh:<2}  desired: {desired}  robot: {rp}  reward: {r:.2f}")
    print()
