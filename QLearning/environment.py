import numpy as np
import time
from q_agent import RandomAgent, QAgent


class Environment:
    def __init__(self):
        self.num_rows = 5
        self.num_cols = 5
        self.num_cells = self.num_cols * self.num_rows
        self.move_probability = 0.2
        self.bomb_positions = (3,3)
        self.gold_positions = (4,3)

        self.rewards = np.zeros(shape = (self.num_rows, self.num_cols), dtype=int)
        self.rewards[self.bomb_positions] = -10
        self.rewards[self.gold_positions] = 10

        self.actions = ("UP", "RIGHT", "DOWN", "LEFT")
        self.num_actions = len(self.actions)


    def place_agent(self):
        self.agent_position = (0, np.random.randint(0,5))
        
    def get_available_actions(self):
        return self.actions

    def make_step(self, action):
        
        if np.random.uniform(0,1) < self.move_probability:
            action_list = list(self.actions)
            action_list.remove(action)
            rand = np.random.randint(self.num_actions)
            action = self.actions[rand]

        old_position = new_position = list(self.agent_position)

        if action == "UP":
            candidate_position = old_position[0] + 1
            if candidate_position < self.num_cols:
                new_position[0] = candidate_position
        elif action == "RIGHT":
            candidate_position = old_position[1] + 1
            if candidate_position < self.num_cols:
                new_position[1] = candidate_position
        elif action == "DOWN":
            candidate_position = old_position[0] - 1
            if candidate_position >= 0:
                new_position[0] = candidate_position
        elif action == "LEFT":
            candidate_position = old_position[1] -1
            if candidate_position >= 0:
                new_position[1] = candidate_position
        else:
            raise ValueError("Action was misspecified!")
        
        self.agent_position = tuple(new_position)

        reward = self.rewards[self.agent_position]
        reward -= 1
        return reward, self.agent_position


if __name__ == "__main__":
    env = Environment()
    agent = RandomAgent()

    while env.agent_position != env.gold_positions and env.agent_position != env.bomb_positions:
        print("Current position of the agent =", env.agent_position)
        available_actions = env.get_available_actions()
        chosen_action = agent.choose_action(available_actions)
        print("Randomly chosen action =", available_actions[chosen_action])
        reward, pos = env.make_step(chosen_action)
        print("Reward obtained =", reward)
        print("Current position of the agent =", pos)
        time.sleep(0.5)