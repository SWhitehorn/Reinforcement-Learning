import time

class SARSA_Agent():
    def __init__(self, track):
        self.track = track
        
        self.learning_rate = 0.2
        self.discount = 0.9
        self.greedy = 0.15
        
        self.states = set()
        self.actions = [i for i in range(9)]
        self.Q = dict()
        self.rewards = []
        
        self.SARSA()
        #self.SARSA_visualise()
        
    def SARSA(self):
        for i in range(150):
            #Start new rewards
            self.rewards.append(0)
            # Reset track and get starting state 
            state = self.track.reset()
            if state not in self.states:
                self.add_state(state)
            action = self.choose_action(state)
            terminal = False
            while not terminal:
                next_state, reward, terminal = self.track.step(action)
                self.rewards[-1] += reward
                if next_state not in self.states:
                    self.add_state(next_state)
                next_action = self.choose_action(next_state)
                
                # Q(S, A) = Q(S, A) + alpha[R + gamma*Q(S',A') - Q(S,A)]
                self.Q[(state, action)] = self.Q[(state, action)] + self.learning_rate*(reward + self.discount*self.Q[(next_state, next_action)] - self.Q[(state, action)])
                state, action = next_state, next_action
    
    def SARSA_visualise(self):
        state = self.track.reset()
        if state not in self.states:
            self.add_state(state)
        action = self.choose_action(state)
        terminal = False
        while not terminal:
            self.track.render()
            next_state, reward, terminal = self.track.step(action)
            self.rewards[-1] += reward
            if next_state not in self.states:
                self.add_state(next_state)
            next_action = self.choose_action(next_state)
            # Q(S, A) = Q(S, A) + alpha[R + gamma*Q(S',A') - Q(S,A)]
            self.Q[(state, action)] = self.Q[(state, action)] + self.learning_rate*(reward + self.discount*self.Q[(next_state, next_action)] - self.Q[(state, action)])
            state, action = next_state, next_action
    
          
    def choose_action(self, state):
        # Agent is greedy
        if np.random.uniform(0,1) > self.greedy:
            action = self.get_best_action(state)
        # Agent chooses to explore
        else:
            # Choose random action
            action = np.random.randint(9) 
        return action
        
    def get_best_action(self, state):
        # Get highest values for action
        best_action = []
        highest_return = float("-inf")
        num_actions = 0
        for action in self.track.get_actions():
            Q_value = self.Q[(state, action)]
            if  Q_value > highest_return:
                best_action = [action]
                highest_return = Q_value
                num_actions = 1
            elif Q_value == highest_return:
                best_action.append(action)
                num_actions+=1
        return best_action[np.random.randint(num_actions)]
    
        
    def add_state(self, state):
        self.states.add(state)
        for action in self.actions:
            self.Q[(state, action)] = 0

start = time.perf_counter()
sarsa_rewards = []
print("Training agents...")
for i in range(50):
    agent = SARSA_Agent(env)
    sarsa_rewards.append(agent.rewards)
end = time.perf_counter()
print(f"Training completed in {end-start:0.4f} seconds")