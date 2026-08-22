"""
MOT5 v11.0 — Smart Equation Discovery Engine
"""
import numpy as np
from .operation_pool import OPERATIONS
from .maha_vault import MahaVault
from .pattern_detector import PatternDetector
from .smart_learner import SmartLearner

class MOT5V9:
    def __init__(self, config=None):
        self.config = config or {}
        self.vault = MahaVault()
        self.pattern_detector = PatternDetector()
        self.learner = SmartLearner()
        self.fitted = False
        self.best_sequence = None
        self.best_pattern = None
        self.verbose = True
    
    def fit(self, X, y, verbose=True):
        self.verbose = verbose
        X = np.array(X)
        y = np.array(y)
        
        if self.verbose:
            print("🔍 Discovering equation...")
        
        pattern = self.pattern_detector.detect(X, y)
        if self.verbose:
            print(f"   📊 Pattern: {pattern}")
        
        start = X[0][0]
        target = y[0]
        
        sequence, _ = self.learner.get_sequence(pattern, start, target)
        
        if not sequence:
            sequence = ['add']
        
        full_sequence = []
        current = start
        for op in sequence:
            param = self.learner.calculate_parameter(current, target, op)
            new_current = self.learner.apply_operation(current, op, param)
            full_sequence.append((op, param))
            current = new_current
        
        pred = self._evaluate_sequence(full_sequence, X)
        error = np.mean((pred - y) ** 2)
        
        reward = 100 if error < 0.01 else (50 if error < 0.1 else 0)
        
        self.learner.learn(pattern, sequence, reward)
        self.vault.add_relation(pattern, sequence)
        
        self.best_pattern = pattern
        self.best_sequence = full_sequence
        self.fitted = True
        
        if self.verbose:
            print(f"   ✅ Equation discovered! Error: {error:.4f}")
        
        return self
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Call fit() first!")
        
        X = np.array(X)
        predictions = []
        for x in X.flatten():
            current = x
            for op, param in self.best_sequence:
                if op in OPERATIONS:
                    if OPERATIONS[op]['arity'] == 1:
                        current = OPERATIONS[op]['func'](current)
                    else:
                        current = OPERATIONS[op]['func'](current, param)
            predictions.append(current)
        return np.array(predictions)
    
    def explain(self):
        """Get human-readable equation"""
        if not self.fitted:
            return {"error": "Call fit() first!"}
        
        if not self.best_sequence:
            return {"error": "No equation found!"}
        
        ops_str = " → ".join([op for op, _ in self.best_sequence])
        params_str = ", ".join([f"{p:.2f}" if p is not None else "None" for _, p in self.best_sequence])
        
        return {
            "equation": ops_str,
            "parameters": params_str,
            "pattern": self.best_pattern,
            "human_readable": f"y = {ops_str} (params: {params_str})"
        }
    
    def _evaluate_sequence(self, sequence, X):
        pred = []
        for x in X.flatten():
            current = x
            for op, param in sequence:
                if op in OPERATIONS:
                    if OPERATIONS[op]['arity'] == 1:
                        current = OPERATIONS[op]['func'](current)
                    else:
                        current = OPERATIONS[op]['func'](current, param)
            pred.append(current)
        return np.array(pred)
    
    def save(self, filepath="maha_vault.json"):
        self.vault.save(filepath)
    
    def load(self, filepath="maha_vault.json"):
        self.vault.load(filepath)
