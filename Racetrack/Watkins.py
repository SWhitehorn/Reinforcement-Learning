import time 

class LambdaAgent():
    def __init__(self, track):
        self.track = track
        
        # Initialise constants
        self.learning_rate = 0.6
        self.discount = 0.9
        self.greedy = 0.15
        self.trace_decay = 0.4
        self.epsilon_decay = 0.001
        
        self.states = set()
        self.actions = [i for i in range(9)]
        
        # Initialise Q(s,a) and e(s, a)
        self.Q = dict()
        self.trace = dict()
        
        self.rewards = []
        
        self.QLearn()
        #self.visualise()
        
    def QLearn(self):
        # Repeat for each episode
        for i in range(150):
            
            #Start new rewards
            self.rewards.append(0)
            
            # Initialise S
            state = self.track.reset()
            if state not in self.states:
                self.add_state(state)
            terminal = False
            
            # Choose A from S using e-greedy policy
            action = self.choose_action(state)
            
            # Repeat until terminal
            while not terminal:     
                
                # Take action, observe R, S'
                next_state, reward, terminal = self.track.step(action)
                self.rewards[-1] += reward
                if next_state not in self.states:
                    self.add_state(next_state)

                # Choose a' from s' using e-greedy policy
                next_action = self.choose_action(next_state)
                
                # a* <- argmax-b Q(s', b)
                a_star = self.get_greedy_action(next_state)
                greedy = (next_action == a_star)
                
                # delta <- r + gamma*Q(s', a') - Q(s, a)
                delta = reward + self.discount * self.Q[(next_state, a_star)] - self.Q[(state, action)]
                 
                # update trace
                self.trace[(state, action)] += 1
                
                # For all s, a
                for s in self.states:
                    for a in self.actions:
                        # Skip traces for which update is minimal for time optimisation
                        if self.trace[(s,a)] > 0.003:
                            
                            # Q(s,a) <- Q(s,a) + alpha * delta * e(s,a)
                            self.Q[(s,a)] = self.Q[(s,a)] + (self.learning_rate * delta * self.trace[(s, a)])
                            self.trace[(s, a)] = self.discount * self.trace_decay * self.trace[(s,a)]
                
                
                state, action = next_state, next_action
            
            # Make policy slightly greedier each episode
            self.greedy -= self.epsilon_decay
                          
    def choose_action(self, state):
        # Agent is greedy
        if np.random.random() < self.greedy:
            return np.random.randint(9) 
        else:
            return self.get_greedy_action(state)
     
    def max_a(self, state):
        # Get highest Q value for state
        highest_return = float("-inf")
        for action in self.actions:
            Q_value = self.Q[(state, action)]
            if  Q_value > highest_return:
                highest_return = Q_value
        return highest_return
        
    
    def get_greedy_action(self, state):
        # Get highest values for action
        best_action = []
        highest_return = float("-inf")
        for action in self.track.get_actions():
            Q_value = self.Q[(state, action)]
            if  Q_value > highest_return:
                best_action = [action]
                highest_return = Q_value
            elif Q_value == highest_return:
                best_action.append(action)
        return np.random.choice(best_action)
    
    # Changed to initialise both Q value for state and trace for state
    def add_state(self, state):
        self.states.add(state)
        for action in self.actions:
            self.Q[(state, action)] = 0
            self.trace[(state, action)] = 0
            
    def visualise(self):
            # Initialise S
            state = self.track.reset()
            if state not in self.states:
                self.add_state(state)
            terminal = False
            
            # Repeat until Terminal
            while not terminal:
                self.track.render()
                # Choose A from S using e-greedy policy
                action = self.choose_action(state)
                
                # Take action, observe R, S'
                next_state, reward, terminal = self.track.step(action)
                if next_state not in self.states:
                    self.add_state(next_state)
                state = next_state
        
        
start = time.perf_counter()
modified_agent_rewards = []
print("Training agents...")
for i in range(25):
    agent = LambdaAgent(env)
    modified_agent_rewards.append(agent.rewards)
end = time.perf_counter()
print(f"Training completed in {end-start:0.4f} seconds")