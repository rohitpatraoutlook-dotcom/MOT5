"""
Test fixed GA
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mot5 import MOT5

print("🔬 Testing Fixed GA")
print("="*50)

np.random.seed(42)
X = np.random.randn(200, 1)
y = 2 * X[:, 0] + 1 + np.random.randn(200) * 0.01

print(f"Data: {len(X)} samples")
print(f"Target: y = 2x + 1 + noise")

model = MOT5({
    'pop_size': 100,
    'generations': 30,
    'dim': 1,
    'max_depth': 4
})

model.fit(X, y)
exp = model.explain()

print(f"\n✅ Discovered: {exp['metric']}")
print(f"📊 Complexity: {exp['complexity']}")

X_test = np.array([[0.0], [1.0], [2.0]])
y_pred = model.predict(X_test)
print(f"\n📈 Predictions: {y_pred}")
print(f"✅ Expected:    [1.0, 3.0, 5.0]")
