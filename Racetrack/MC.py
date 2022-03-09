class MCAgent():
    def __init__(self, track):     
        self.track = track

        # Initialise all states, value estimates, returns.
        # Initalisations are implicit and are only formally added when a state has been reached.
        self.states = set()
        self.actions = [i for i in range(9)]
        self.policy = dict()
        self.Q = dict()
        self.returns = dict()
        self.rewards = []
        
        self.discount_factor = 0.9
        self.explore = 0.15
        self.full_monte()
        
    def full_monte(self):
        # Repeat for each episode
        for i in range(150):
            self.rewards.append(0)
            history = self.generate_episode()
            G = 0 
            self.last_pair = history[-1][0]
            #Iterate backwards over array
            for j in range(len(history)-1, -1, -1): 
                SA_pair, reward = history[j]
                self.rewards[-1] += reward
                G = G*self.discount_factor + reward
                
                if self.first_visit(SA_pair, j, history):
                    self.returns[SA_pair] = np.append(self.returns[SA_pair],G)
                    self.Q[SA_pair] = np.average(self.returns[SA_pair])
                    A = self.get_highest_value(SA_pair[0])                    
                    
                    for pos_action in self.actions:
                        if pos_action == A:
                            self.policy[(SA_pair[0], pos_action)] = (1 - self.explore + self.explore/len(self.actions))         
                        else:
                            self.policy[(SA_pair[0], pos_action)] = self.explore/len(self.actions)
    
    def generate_episode(self):
        state = self.track.reset()
        if state not in self.states:
            self.add_state(state)  
        episode_history = []
        while True:
            # Agent is greedy
            if np.random.uniform(0,1) > self.explore:
                best_action = self.get_best_action_policy(state)
            
            # Agent chooses to explore
            else:
                # Choose random action
                best_action = np.random.randint(9)       
            
            next_state, reward, terminal = self.track.step(best_action)
            if next_state not in self.states:
                self.add_state(next_state) 
            episode_history.append(((state, best_action), reward))
            state = next_state
            if terminal:
                return episode_history

    def first_visit(self, pair, limit, history):
        for i in range(limit):
            if history[i][0] == pair:
                return False
        return True

    def get_highest_value(self, state):
        best_action = [0]
        highest_return = float("-inf")
        num_actions = 0
        for action in self.actions:
            expected_value = self.Q[(state, action)]
            if  expected_value > highest_return:
                best_action = [action]
                highest_return = expected_value
                num_actions = 1
            elif expected_value == highest_return:
                best_action.append(action)
                num_actions+=1
        return best_action[np.random.randint(num_actions)]
            
    
    def get_best_action_policy(self, state):
        # Get highest values for action 
        best_action = [0]
        highest_return = 0
        num_actions = 0
        for action in self.track.get_actions():
            policy_value = self.policy[(state, action)]
            if  policy_value > highest_return:
                best_action = [action]
                highest_return = policy_value
                num_actions = 1
            elif policy_value == highest_return:
                best_action.append(action)
                num_actions+=1
        return best_action[np.random.randint(num_actions)]
    
    # Allow for optimisisation by only making SA pair explicit when it has been reached
    def add_state(self, state):
        self.states.add(state)
        for action in self.actions:
            self.policy[(state, action)] = 0.125
            self.Q[(state, action)] = 0
            self.returns[(state, action)] = 0
        

start = time.perf_counter()
mc_rewards = []
print("Training agents...")
for i in range(50):
    mc = MCAgent(env)
    mc_rewards.append(mc.rewards)
end = time.perf_counter()
print(f"Training completed in {end-start:0.4f} seconds")