"""
Sequence Predictor — Smarter Q-Learning
"""
import numpy as np
import random
from .operation_pool import OPERATIONS, OPERATION_NAMES

class SequencePredictor:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
    
    def get_state(self, current, target, pattern):
        diff = abs(target - current)
        return f"{pattern}_{int(current)}_{int(target)}_{int(diff)}"
    
    def get_actions(self, current, target):
        """Smart actions based on current value"""
        actions = []
        
        # If target > current: prefer add/mul
        if target > current:
            actions.extend([('add', 1), ('add', 2), ('add', 5), ('mul', 2), ('mul', 3), ('mul', 5)])
        # If target < current: prefer sub/div
        else:
            actions.extend([('sub', 1), ('sub', 2), ('sub', 5), ('div', 2), ('div', 3), ('div', 5)])
        
        # Always include unary ops
        for op in ['sin', 'cos', 'exp', 'log', 'abs', 'sqrt']:
            actions.append((op, None))
        
        return actions
    
    def predict(self, current, target, pattern):
        """Predict next operation"""
        state = self.get_state(current, target, pattern)
        actions = self.get_actions(current, target)
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Exploitation
        if state in self.q_table:
            best_action = max(self.q_table[state], key=self.q_table[state].get)
            return best_action
        else:
            return random.choice(actions)
    
    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        
        old_value = self.q_table[state].get(action, 0)
        max_future = max(self.q_table.get(next_state, {}).values()) if next_state in self.q_table else 0
        new_value = old_value + self.alpha * (reward + self.gamma * max_future - old_value)
        self.q_table[state][action] = new_value
    
    def get_sequence(self, start, target, pattern, max_steps=10):
        sequence = []
        current = start
        step_count = 0
        
        while step_count < max_steps:
            action = self.predict(current, target, pattern)
            sequence.append(action)
            
            op, param = action
            if op in OPERATIONS:
                if OPERATIONS[op]['arity'] == 1:
                    current = OPERATIONS[op]['func'](current)
                else:
                    current = OPERATIONS[op]['func'](current, param)
            
            step_count += 1
            
            if abs(current - target) < 1e-6:
                break
        
        return sequence
