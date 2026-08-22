"""
TEST 1: Vector Output (Multiple y's)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from mot5_pro import MOT5Pro

print("="*60)
print("🧪 TEST 1: Vector Output (Multi-Output)")
print("="*60)

np.random.seed(42)
X = np.random.randn(150, 2)
y1 = 2*X[:,0] + 3*X[:,1] + 1 + np.random.randn(150) * 0.01
y2 = np.sin(X[:,0]) + np.cos(X[:,1]) + np.random.randn(150) * 0.01
Y = np.column_stack([y1, y2])

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Outputs: 2 (y1, y2)")

# Train
model = MOT5Pro(n_outputs=2)
model.fit(X, Y)

# Test
X_test = np.random.randn(5, 2)
y_pred = model.predict(X_test)
print(f"\n✅ Predictions:\n{y_pred}")

# Explain
for i in range(2):
    exp = model.explain(i)
    print(f"Output {i+1}: {exp['metric']}")

print("\n✅ TEST 1 COMPLETE!")
