"""
Pattern Detector — Improved for Complex Functions
"""
import numpy as np

class PatternDetector:
    def __init__(self):
        self.patterns = ['linear', 'sin', 'cos', 'exp', 'complex']
    
    def detect(self, X, y):
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        n = len(X)
        
        # 1. Linear check
        if n > 2:
            dx = X[1:] - X[:-1]
            dy = y[1:] - y[:-1]
            ratio = dy / (dx + 1e-10)
            if np.std(ratio) < 0.05:
                return 'linear'
        
        # 2. Sin check
        for freq in [0.5, 1.0, 2.0, 3.0]:
            pred = np.sin(freq * X)
            if np.mean((y - pred)**2) < 0.1:
                return 'sin'
        
        # 3. Cos check
        for freq in [0.5, 1.0, 2.0, 3.0]:
            pred = np.cos(freq * X)
            if np.mean((y - pred)**2) < 0.1:
                return 'cos'
        
        # 4. Exp check
        if n > 3 and np.all(y > 0):
            log_y = np.log(np.abs(y) + 1e-10)
            dx = X[1:] - X[:-1]
            dlog = log_y[1:] - log_y[:-1]
            if np.std(dlog / (dx + 1e-10)) < 0.1:
                return 'exp'
        
        # 5. Default complex
        return 'complex'
