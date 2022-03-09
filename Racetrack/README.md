Three implementations of reinforcement learning algorithms to solve an environment based on a racetrack. 

The environment is defined as a 2D grid, with squares being either start squares, part of the track, a terminal square, or off track. The agent receives a reward of -1 for each move it makes, and -10 for running off the track (which will reset it to a random starting square). The agent receives a reward of 10 for successfully reaching the finish line, at which point the episode ends. Unfortunately, the code defining the environment was not written by me, so I am unable to include it here.

Each file trains 20 agents on 150 episodes. The rewards earned during training are stored in a rewards list, which was used to produce a plot for comparison. 
