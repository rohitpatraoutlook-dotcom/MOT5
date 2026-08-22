"""
Simple test: Discover polynomial from data
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

# Generate data: y = 2x + 1 (with noise)
np.random.seed(42)
X = np.random.randn(200, 1)
y = 2 * X[:, 0] + 1 + np.random.randn(200) * 0.1

print(f"Data: {X.shape[0]} samples, {X.shape[1]} feature")
print(f"Target: y = 2x + 1 + noise")

# Fit MOT5
model = MOT5({
    'pop_size': 50,
    'generations': 20,
    'dim': 1,
    'max_depth': 3
})

model.fit(X, y)

# Explain
print("\n" + "="*50)
print("DISCOVERED EQUATION")
print("="*50)
exp = model.explain()
print(f"Equation: y = {exp['metric']}")
print(f"Complexity: {exp['complexity']}")

# Test prediction
X_test = np.array([[0.0], [1.0], [2.0], [-1.0]])
y_pred = model.predict(X_test)
print(f"\nPredictions at x={[0,1,2,-1]}: {y_pred}")

print("\n✅ Done!")
