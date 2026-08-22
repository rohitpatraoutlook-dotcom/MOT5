"""
TEST 1 V2: Vector Output with Aggressive Settings
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from mot5 import MOT5

print("="*60)
print("🧪 TEST 1 V2: Vector Output (Aggressive)")
print("="*60)

np.random.seed(42)
X = np.random.randn(300, 2)  # More data
y1 = 2*X[:,0] + 3*X[:,1] + 1 + np.random.randn(300) * 0.01
y2 = np.sin(X[:,0]) + np.cos(X[:,1]) + np.random.randn(300) * 0.01

print(f"Data: {len(X)} samples, {X.shape[1]} features")
print(f"Targets: y1 = 2x0 + 3x1 + 1, y2 = sin(x0) + cos(x1)")

# Train output 1 with aggressive settings
print("\n📚 Training Output 1...")
model1 = MOT5({
    'pop_size': 150,
    'generations': 60,
    'dim': 2,
    'max_depth': 4
})
model1.fit(X, y1)

# Train output 2 with aggressive settings
print("📚 Training Output 2...")
model2 = MOT5({
    'pop_size': 150,
    'generations': 60,
    'dim': 2,
    'max_depth': 4
})
model2.fit(X, y2)

# Results
exp1 = model1.explain()
exp2 = model2.explain()

print(f"\n✅ Output 1: {exp1['metric']}")
print(f"   Complexity: {exp1['complexity']}")
print(f"\n✅ Output 2: {exp2['metric']}")
print(f"   Complexity: {exp2['complexity']}")

# Test
X_test = np.random.randn(5, 2)
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)

print(f"\n📈 Sample predictions:")
print(f"   X: {X_test[:2]}")
print(f"   y1: {y_pred1[:2]}")
print(f"   y2: {y_pred2[:2]}")

print("\n✅ TEST 1 V2 COMPLETE!")
