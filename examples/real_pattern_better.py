"""
Better MOT5 on real patterns with more data/generations
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

patterns = {
    'sigmoid': lambda x: 1/(1 + np.exp(-x)),
    'sin_wave': lambda x: np.sin(x),
    'exp_decay': lambda x: np.exp(-x**2/10)
}

np.random.seed(42)

for name, func in patterns.items():
    print(f"\n🔍 Pattern: {name}")
    print("-"*40)
    
    # More data, wider range
    X = np.linspace(-5, 5, 500).reshape(-1, 1)
    y = func(X[:, 0]) + np.random.randn(500) * 0.05
    
    # More powerful config
    model = MOT5({
        'pop_size': 100,
        'generations': 50,
        'dim': 1,
        'max_depth': 5
    })
    
    model.fit(X, y)
    exp = model.explain()
    print(f"  Found: y = {exp['metric']}")
    print(f"  Complexity: {exp['complexity']}")
    
    # Test accuracy
    X_test = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y_pred = model.predict(X_test)
    y_actual = func(X_test[:, 0])
    
    mse = np.mean((y_pred - y_actual)**2)
    print(f"  Test MSE: {mse:.4f}")
