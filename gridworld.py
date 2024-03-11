import numpy as np
class GridWorld:
    def __init__(self, n_rows, n_cols, goal, bad, wall):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.goal = goal
        self.bad = bad
        self.wall = wall
        self.actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        self.V = np.zeros((n_rows, n_cols))
        self.V[self.goal] = 1
        self.V[self.bad] = -1

    def move(self, state, action):
        x, y = state
        if action == (1, 0):
            return (x + 1, y)
        elif action == (-1, 0):
            return (x - 1, y)
        elif action == (0, 1):
            return (x, y + 1)
        elif action == (0, -1):
            return (x, y - 1)

    def is_valid_move(self, position, action):
        new_position = self.move(position, action)
        x, y = new_position
        if 0 <= x < self.n_rows and 0 <= y < self.n_cols and new_position != self.wall:
            return True
        return False

    def possible_actions(self, position):
        valid_actions = []
        if position != self.wall:
            for action in self.actions:
                if self.is_valid_move(position, action):
                    valid_actions.append(action)
        return valid_actions

    def action_90degree(self, action, actions_possible):
        actions = []
        for i in actions_possible:
            if i != action:
                if i + action != (0, 0):
                    actions.append(i)
        return actions

    def get_state_index(self, x, y):
        return x * self.n_cols + y
    
    def iterate_value(self):
        delta = 0
        for x in range(self.n_rows):
            for y in range(self.n_cols):
                if (x, y) == self.wall or (x, y) == self.goal or (x, y) == self.bad:
                    continue  # Pas de calcul pour ces cases
                v = self.V[x, y]
                Vs = []
                pa = self.possible_actions((x, y))
                for action in pa:
                    sum_derivations = 0
                    for other_action in self.action_90degree(action, pa):
                        proba = 0.2 if len(self.action_90degree(action, pa)) == 1 else 0.1
                        sum_derivations += proba * self.V[*self.move((x, y), other_action)]
                    sum_action = self.reward[x, y] + self.gamma * 0.8 * self.V[*self.move((x, y), action)]
                    sum_rewards = sum_action + sum_derivations
                    Vs.append(sum_rewards)
                self.V[x, y] = max(Vs)
                delta = max(delta, abs(v - self.V[x, y]))
        return delta

