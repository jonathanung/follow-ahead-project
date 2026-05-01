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

    def ucb(self, c=2.0):
        if self.parent is None or self.parent.visits == 0:
            return float('inf')
        eps = 1e-8
        exploration = c * math.sqrt(math.log(self.parent.visits) / (self.visits + eps))
        return (self.q() + exploration) * self.prior

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, c=2.0):
        if not self.children:
            return None
        def ucb_with_tiebreak(n):
            # Prefer 'straight' action in ties if it exists
            score = n.ucb(c)
            tie_break = 1 if n.action in ['straight', 'fast_straight'] else 0
            return (score, tie_break)
        return max(self.children, key=ucb_with_tiebreak)

    def best_action_child(self):
        if not self.children:
            return None
        def visits_with_tiebreak(n):
            # Prefer 'straight' action in ties if it exists
            score = n.visits
            tie_break = 1 if n.action in ['straight', 'fast_straight'] else 0
            return (score, tie_break)
        return max(self.children, key=visits_with_tiebreak)