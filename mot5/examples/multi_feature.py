"""
Test MOT5 on multi-feature data
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

np.random.seed(42)
n = 200

# y = 2*x1 + 3*x2 + 1
X = np.random.randn(n, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + np.random.randn(n) * 0.1

print(f"Data: {n} samples, 2 features")
print(f"Target: y = 2x1 + 3x2 + 1 + noise")

model = MOT5({
    'pop_size': 80,
    'generations': 30,
    'dim': 2,
    'max_depth': 4
})

model.fit(X, y)

exp = model.explain()
print(f"\n✅ Discovered: y = {exp['metric']}")

# Test predictions
X_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_pred = model.predict(X_test)
y_actual = 2*X_test[:,0] + 3*X_test[:,1] + 1

print("\nPredictions:")
for i in range(len(X_test)):
    print(f"  x=({X_test[i,0]}, {X_test[i,1]}) → predicted: {y_pred[i]:.3f}, actual: {y_actual[i]:.3f}")
