import math

class Node:
    def __init__(self, state, parent=None, action=None, prior=1.0, node_type='robot'):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.node_type = node_type
        self.children = []
        self.visits = 0
        self.total_value = 0.0

    def q(self):
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb(self, c=1.4):
        if self.parent is None or self.parent.visits == 0:
            return float('inf')
        eps = 1e-8
        return self.q() + c * self.prior * math.sqrt(math.log(self.parent.visits) / (self.visits + eps))

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self):
        return max(self.children, key=lambda n: n.ucb())

    def best_action_child(self):
        return max(self.children, key=lambda n: n.visits)
