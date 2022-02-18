import numpy as np

class RandomAgent:
    def choose_action(self, actions):
        num_actions = len(actions)
        action_index = np.random.randint(0, num_actions)
        return action_index

class QAgent:
    def __init__(self):
        self.learning_rate = 0.1
        self.greedy_policy = 0.05
        self.QTable = {}
        self.default  = 0
        

    def choose_action(self, state, actions):
        if np.random.uniform(0,1) < self.greedy_policy:
            return actions[np.random.randint(0, len(actions))]
        
        payoff = float("-inf")
        best_action = None

        for action in actions:
            action_reward = self.QTable[(state, action)]
            if action_reward > payoff:
                payoff = action_reward
                best_action = action
        return best_action

    def learn(self, move, state, actions, reward):
        self.QTable[move] = (1-self.learning_rate)*self.QTable[move] + self.learning_rate*(reward + max(
            self.QTable[(state, "UP")], 
            self.QTable[(state, "RIGHT")],
            self.QTable[(state, "DOWN")],
            self.QTable[(state, "LEFT")])
            )

    def add_Q_values(self, state, actions):
        for action in actions:
            if (state, action) not in self.QTable:
                self.QTable[(state, action)] = self.default