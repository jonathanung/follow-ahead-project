import time
import random
import copy
from node import Node
from simple_grid import transition, reward, ROBOT_ACTIONS
from human_model import human_probabilities
from rollout import rollout

MAX_DEPTH = 20

def _expand_robot(node):
    p = 1.0 / len(ROBOT_ACTIONS)
    for action in ROBOT_ACTIONS:
        child = Node(copy.deepcopy(node.state), parent=node, action=action, prior=p, node_type='human')
        node.children.append(child)

def _expand_human(node):
    probs = human_probabilities(node.state)
    for action, p in probs.items():
        new_state = transition(node.state, node.action, action)
        child = Node(new_state, parent=node, action=action, prior=p, node_type='robot')
        node.children.append(child)

def _select(node):
    depth = 0
    while not node.is_leaf():
        if depth >= MAX_DEPTH:
            break
        node = node.best_child()
        depth += 1
    return node

def _backprop(node, value):
    while node is not None:
        node.visits += 1
        node.total_value += value
        node = node.parent

def _run_tree(root, budget_fn):
    while budget_fn():
        leaf = _select(root)
        if leaf.visits == 0:
            value = reward(leaf.state) + rollout(leaf.state)
            _backprop(leaf, value)
            continue
        if leaf.node_type == 'robot':
            _expand_robot(leaf)
        else:
            _expand_human(leaf)
        if leaf.children:
            leaf = random.choice(leaf.children)
        value = reward(leaf.state) + rollout(leaf.state)
        _backprop(leaf, value)


class MCTSPlanner:
    def __init__(self, n_simulations=None, time_budget=None, verbose=False):
        self.n_simulations = n_simulations
        self.time_budget = time_budget
        self.verbose = verbose
        if n_simulations is None and time_budget is None:
            self.n_simulations = 1000

    def plan(self, state):
        root = Node(copy.deepcopy(state), node_type='robot')

        if self.time_budget is not None:
            deadline = time.time() + self.time_budget
            budget_fn = lambda: time.time() < deadline
        else:
            counter = [0]
            limit = self.n_simulations
            def budget_fn():
                counter[0] += 1
                return counter[0] <= limit

        _run_tree(root, budget_fn)

        if self.verbose:
            print(f"root visits: {root.visits}")
            print(f"children: {[(c.action, c.visits) for c in root.children]}")

        return root.best_action_child().action
