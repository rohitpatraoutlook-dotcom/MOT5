"""
Discover clean polynomial: y = 2x + 1 (no noise)
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

# Clean data: y = 2x + 1 (NO NOISE)
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X[:, 0] + 1

print(f"Target: y = 2x + 1 (NO NOISE)")

model = MOT5({
    'pop_size': 80,
    'generations': 30,
    'dim': 1,
    'max_depth': 3
})

model.fit(X, y)

exp = model.explain()
print(f"\nEquation: y = {exp['metric']}")
print(f"Complexity: {exp['complexity']}")

X_test = np.array([[0.0], [1.0], [2.0]])
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred}")
