"""
Find best hyperparameters
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X[:, 0] + 1 + np.random.randn(100) * 0.1

# Test different configs
configs = [
    {'pop_size': 30, 'generations': 10},
    {'pop_size': 50, 'generations': 20},
    {'pop_size': 80, 'generations': 30},
    {'pop_size': 100, 'generations': 50},
]

print("HYPERPARAMETER TUNING")
print("="*50)

for cfg in configs:
    print(f"\nTesting: pop={cfg['pop_size']}, gens={cfg['generations']}")
    model = MOT5(cfg)
    model.fit(X, y)
    exp = model.explain()
    print(f"  Equation: y = {exp['metric']}")
    print(f"  Complexity: {exp['complexity']}")
