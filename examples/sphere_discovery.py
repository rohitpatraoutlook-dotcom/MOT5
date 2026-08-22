"""
Example: Discover sphere metric from data
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

# Generate sphere data
def generate_sphere(n=500, radius=1.0):
    """Generate points on a sphere"""
    theta = np.random.uniform(0, np.pi, n)
    phi = np.random.uniform(0, 2*np.pi, n)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    return np.column_stack([x, y, z])

# Generate data
X = generate_sphere(500)
y = np.sum(X ** 2, axis=1)  # Distance from center

print(f"Data shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Fit MOT5
model = MOT5({'pop_size': 50, 'generations': 20})
model.fit(X, y)

# Explain
print("\n" + "="*50)
print("DISCOVERED METRIC")
print("="*50)
print(model.explain())

# Predict
X_test = generate_sphere(50)
y_pred = model.predict(X_test)

print(f"\nPrediction shape: {y_pred.shape}")
print(f"Sample prediction: {y_pred[:5]}")
