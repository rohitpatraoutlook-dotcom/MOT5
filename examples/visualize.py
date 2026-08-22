"""
Visualize MOT5 predictions
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
import matplotlib.pyplot as plt
from mot5 import MOT5

# Generate data
np.random.seed(42)
X = np.random.uniform(-2, 2, 100).reshape(-1, 1)
y = 2 * X[:, 0] + 1 + np.random.randn(100) * 0.1

# Fit model
model = MOT5({'pop_size': 80, 'generations': 30, 'dim': 1})
model.fit(X, y)

# Predict
X_test = np.linspace(-2, 2, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label='Data')
plt.plot(X_test[:, 0], y_pred, 'r-', linewidth=2, label='MOT5 Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f"Discovered: y = {model.explain()['metric']}")
plt.legend()
plt.grid(True, alpha=0.3)

# Save
plt.savefig('mot5_result.png')
print("✅ Plot saved as: mot5_result.png")
print(f"Discovered equation: y = {model.explain()['metric']}")
