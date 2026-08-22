"""
Smart Learner — Safe + Learned
"""
import numpy as np
from .operation_pool import OPERATIONS, OPERATION_NAMES

class SmartLearner:
    def __init__(self):
        self.q_table = {}
        self.history = []
        self._seed_knowledge()
    
    def _seed_knowledge(self):
        self.q_table["linear_0_mul"] = 10
        self.q_table["linear_1_add"] = 8
        self.q_table["sin_0_sin"] = 10
        self.q_table["cos_0_cos"] = 10
        self.q_table["exp_0_exp"] = 10
    
    def get_state(self, pattern, step, current, target):
        return f"{pattern}_{step}"
    
    def decide_operation(self, pattern, step, current, target):
        state = self.get_state(pattern, step, current, target)
        
        best_op = None
        best_score = -float('inf')
        
        for op in OPERATION_NAMES:
            key = f"{state}_{op}"
            score = self.q_table.get(key, 0)
            if score > best_score:
                best_score = score
                best_op = op
        
        # 🔥 SIN FIX: Agar pattern sin hai, to force sin use karo
        if pattern == 'sin' and step == 0:
            return 'sin'
        
        if best_op is None:
            best_op = self._safe_default(current, target)
        
        return best_op
    
    def _safe_default(self, current, target):
        diff = target - current
        if diff > 5: return 'mul'
        elif diff > 1: return 'add'
        elif diff < -5: return 'div'
        elif diff < -1: return 'sub'
        else: return 'add'
    
    def calculate_parameter(self, current, target, op):
        if op == 'add': return target - current
        elif op == 'sub': return current - target
        elif op == 'mul': return target / current if current != 0 else 1
        elif op == 'div': return current / target if target != 0 else 1
        else: return 1
    
    def apply_operation(self, current, op, param):
        if op not in OPERATIONS: return current
        try:
            if OPERATIONS[op]['arity'] == 1:
                result = OPERATIONS[op]['func'](current)
            else:
                result = OPERATIONS[op]['func'](current, param)
            if np.isnan(result) or np.isinf(result): return current
            return np.clip(result, -1e6, 1e6)
        except: return current
    
    def learn(self, pattern, sequence, reward):
        for step, op in enumerate(sequence):
            state = f"{pattern}_{step}"
            key = f"{state}_{op}"
            old_value = self.q_table.get(key, 0)
            self.q_table[key] = old_value + 0.1 * (reward - old_value)
    
    def get_sequence(self, pattern, start, target, max_steps=10):
        sequence = []
        current = start
        step = 0
        
        print(f"   🧠 Smart Learning: {start} → {target}")
        
        while step < max_steps and abs(current - target) > 0.001:
            op = self.decide_operation(pattern, step, current, target)
            param = self.calculate_parameter(current, target, op)
            new_current = self.apply_operation(current, op, param)
            
            if abs(new_current - current) < 0.001:
                break
            
            sequence.append(op)
            current = new_current
            step += 1
            print(f"      Step {step}: {op}({param:.2f}) → {current:.4f}")
        
        # 🔥 SIN FIX: Agar sequence empty hai aur pattern sin hai
        if not sequence and pattern == 'sin':
            sequence = ['sin']
            current = np.sin(start)
            print(f"      Step 1: sin → {current:.4f}")
        
        return sequence, current
