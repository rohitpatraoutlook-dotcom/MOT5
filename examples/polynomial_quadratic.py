"""
Discover quadratic: y = 3x² + 2x + 1
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

np.random.seed(42)
X = np.random.randn(200, 1)
y = 3 * X[:, 0]**2 + 2 * X[:, 0] + 1 + np.random.randn(200) * 0.1

print(f"Target: y = 3x² + 2x + 1 + noise")

model = MOT5({
    'pop_size': 100,
    'generations': 40,
    'dim': 1,
    'max_depth': 4
})

model.fit(X, y)

exp = model.explain()
print(f"\nEquation: y = {exp['metric']}")
print(f"Complexity: {exp['complexity']}")

X_test = np.array([[0.0], [1.0], [2.0], [-1.0]])
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred}")
