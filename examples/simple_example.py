"""
MOT5: 3 Lines to Discover Equation! (Final Working)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mot5 import MOT5

# 1. More data for better discovery
np.random.seed(42)
X = np.random.randn(200, 1)  # 200 samples
y = 2 * X[:, 0] + 1 + np.random.randn(200) * 0.05

print(f"📊 Data: {len(X)} samples, {X.shape[1]} feature")
print(f"🎯 Target: y = 2x + 1 + noise")

# 2. Discover with good settings
model = MOT5({
    'pop_size': 80,
    'generations': 30,
    'max_depth': 4,
    'dim': 1
})

model.fit(X, y)

# 3. Done!
exp = model.explain()
print(f"\n✅ Equation: {exp['metric']}")
print(f"📊 Complexity: {exp['complexity']}")

# Test prediction
X_test = np.array([[0.0], [1.0], [2.0]])
y_pred = model.predict(X_test)
print(f"\n📈 Predictions: {y_pred}")
print(f"✅ Expected:    [1.0, 3.0, 5.0]")
