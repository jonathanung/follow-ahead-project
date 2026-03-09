import copy
import random
from node import Node
from simple_grid import transition, reward, ROBOT_ACTIONS
from human_model import human_probabilities
from rollout import rollout

def expand_robot(node):
    p = 1.0 / len(ROBOT_ACTIONS)
    for action in ROBOT_ACTIONS:
        child_state = copy.deepcopy(node.state)
        child = Node(child_state, parent=node, action=action, prior=p, node_type='human')
        node.children.append(child)

def expand_human(node):
    probs = human_probabilities(node.state)
    for action, p in probs.items():
        new_state = transition(node.state, node.action, action)
        child = Node(new_state, parent=node, action=action, prior=p, node_type='robot')
        node.children.append(child)

def select(node):
    while not node.is_leaf():
        node = node.best_child()
    return node

def backprop(node, value):
    while node is not None:
        node.visits += 1
        node.total_value += value
        node = node.parent

def mcts(state, n_simulations=500, rollout_depth=10):
    root = Node(state, node_type='robot')

    for _ in range(n_simulations):
        leaf = select(root)

        if leaf.visits == 0:
            value = reward(leaf.state) + rollout(leaf.state, depth=rollout_depth)
            backprop(leaf, value)
            continue

        if leaf.node_type == 'robot':
            expand_robot(leaf)
        else:
            expand_human(leaf)

        if leaf.children:
            leaf = random.choice(leaf.children)

        value = reward(leaf.state) + rollout(leaf.state, depth=rollout_depth)
        backprop(leaf, value)

    best = root.best_action_child()
    return best.action
