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

    return {'robot_pos': new_rp, 'human_pos': new_hp, 'human_heading': new_hh}

def reward(state):
    desired = desired_robot_pos(state['human_pos'], state['human_heading'])
    rp = state['robot_pos']
    dist = ((rp[0] - desired[0])**2 + (rp[1] - desired[1])**2)**0.5
    r = -(dist ** 2)
    if rp == state['human_pos']:
        r -= 10
    return r

def render(state):
    grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    hx, hy = state['human_pos']
    rx, ry = state['robot_pos']
    grid[hy][hx] = 'H'
    if (rx, ry) != (hx, hy):
        grid[ry][rx] = 'R'
    for row in grid:
        print(' '.join(row))
    print(f"Heading: {state['human_heading']}")
    print()
