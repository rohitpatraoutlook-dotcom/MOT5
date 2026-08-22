"""
Test MOT5 on real-world patterns
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

patterns = {
    'sales': lambda x: 100 + 50*np.sin(x*0.1) + 2*x,
    'physics': lambda x: 1/(1 + np.exp(-x)),
    'finance': lambda x: np.exp(0.01*x) + 0.5*np.sin(x)
}

np.random.seed(42)

for name, func in patterns.items():
    print(f"\n🔍 Pattern: {name}")
    print("-"*40)
    
    X = np.linspace(-10, 10, 200).reshape(-1, 1)
    y = func(X[:, 0]) + np.random.randn(200) * 0.5
    
    model = MOT5({
        'pop_size': 60,
        'generations': 25,
        'dim': 1,
        'max_depth': 4
    })
    
    model.fit(X, y)
    exp = model.explain()
    print(f"  Found: y = {exp['metric']}")
    print(f"  Complexity: {exp['complexity']}")
