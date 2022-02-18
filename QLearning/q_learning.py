import numpy as np
from environment import Environment
from q_agent import QAgent
import matplotlib.pyplot as plt






env = Environment()
agent = QAgent()
actions = env.get_available_actions()
samples = []

for i in range(1000):
    steps = 0
    env.place_agent()
    agent.add_Q_values(env.agent_position, actions)
    while env.agent_position != env.gold_positions and env.agent_position != env.bomb_positions:
        old_position = env.agent_position
        chosen_action = agent.choose_action(env.agent_position, actions)
        reward, state = env.make_step(chosen_action)
        agent.add_Q_values(state, actions)
        agent.learn((old_position, chosen_action), state, actions, reward)
        steps += 1
    samples.append(steps)
        
        
fig = plt.plot(samples)
plt.show()