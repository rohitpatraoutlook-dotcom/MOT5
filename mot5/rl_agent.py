"""
RL Agent — Reinforcement Learning for Sequence Optimization
"""
import numpy as np
import random
from collections import deque

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 0.01
        self.q_table = {}
    
    def get_state_key(self, state):
        """Convert state array to string key"""
        return str(state.tolist())
    
    def act(self, state):
        """Choose action using epsilon-greedy"""
        key = self.get_state_key(state)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        if key in self.q_table:
            return np.argmax(self.q_table[key])
        else:
            return random.randint(0, self.action_size - 1)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Experience replay"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            key = self.get_state_key(state)
            next_key = self.get_state_key(next_state)
            
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.action_size)
            
            if done:
                target = reward
            else:
                if next_key in self.q_table:
                    target = reward + self.gamma * np.max(self.q_table[next_key])
                else:
                    target = reward
            
            self.q_table[key][action] += self.alpha * (target - self.q_table[key][action])
    
    def save(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f)
    
    def load(self, filepath):
        import json
        try:
            with open(filepath, 'r') as f:
                self.q_table = json.load(f)
        except:
            pass
