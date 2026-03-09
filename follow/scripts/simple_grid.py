GRID_SIZE = 7
DIRECTIONS = ['N', 'E', 'S', 'W']
DIR_VECTORS = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
ROBOT_ACTIONS = ['N', 'S', 'E', 'W']
HUMAN_ACTIONS = ['forward', 'left', 'right']

OBSTACLES = {(1, 3), (2, 3), (1, 4), (1, 5)}

def is_valid(pos):
    return pos not in OBSTACLES

def turn_left(heading):
    return DIRECTIONS[(DIRECTIONS.index(heading) - 1) % 4]

def turn_right(heading):
    return DIRECTIONS[(DIRECTIONS.index(heading) + 1) % 4]

def move_in_dir(pos, direction):
    dx, dy = DIR_VECTORS[direction]
    nx = max(0, min(GRID_SIZE - 1, pos[0] + dx))
    ny = max(0, min(GRID_SIZE - 1, pos[1] + dy))
    if (nx, ny) in OBSTACLES:
        return pos
    return (nx, ny)

def desired_robot_pos(human_pos, human_heading):
    candidate = move_in_dir(human_pos, human_heading)
    if candidate != human_pos:
        return candidate
    for d in DIRECTIONS:
        alt = move_in_dir(human_pos, d)
        if alt != human_pos:
            return alt
    return human_pos

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

    new_rp = move_in_dir(rp, robot_action)
    if new_rp == new_hp:
        new_rp = rp

    return {
        'robot_pos': new_rp,
        'human_pos': new_hp,
        'human_heading': new_hh,
    }

def reward(state):
    desired = desired_robot_pos(state['human_pos'], state['human_heading'])
    rp = state['robot_pos']
    dist = ((rp[0] - desired[0])**2 + (rp[1] - desired[1])**2)**0.5
    return -(dist ** 2)

def render(state, step=None, robot_action=None, human_action=None):
    hp = state['human_pos']
    rp = state['robot_pos']
    hh = state['human_heading']
    desired = desired_robot_pos(hp, hh)
    r = reward(state)

    grid_rows, heat_rows = [], []
    for y in range(GRID_SIZE):
        grow, hrow = [], []
        for x in range(GRID_SIZE):
            if (x, y) in OBSTACLES:
                grow.append('#')
            elif (x, y) == hp:
                grow.append('H')
            elif (x, y) == rp:
                grow.append('R')
            else:
                grow.append('.')

            if (x, y) in OBSTACLES:
                hrow.append('#')
            elif (x, y) == desired:
                hrow.append('*')
            elif (x, y) == hp:
                hrow.append('H')
            elif (x, y) == rp:
                hrow.append('R')
            else:
                dist = ((x - desired[0])**2 + (y - desired[1])**2)**0.5
                max_dist = ((GRID_SIZE - 1)**2 + (GRID_SIZE - 1)**2)**0.5
                hrow.append(str(int((1 - dist / max_dist) * 9)))
        grid_rows.append(' '.join(grow))
        heat_rows.append(' '.join(hrow))

    if step is not None:
        header = f"step {step}"
        if robot_action and human_action:
            header += f"  |  robot: {robot_action:<8} human: {human_action}"
        print(header)

    for g, h in zip(grid_rows, heat_rows):
        print(g + '     ' + h)

    print(f"heading: {hh:<2}  desired: {desired}  robot: {rp}  reward: {r:.2f}\n")
