"""
Custom function discovery
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

# Try different functions
np.random.seed(42)
n = 100

# Choose function
print("Choose function:")
print("1: y = sin(x)")
print("2: y = exp(-x²)")
print("3: y = x³ - x")
choice = input("Enter choice (1-3): ")

X = np.random.uniform(-2, 2, n).reshape(-1, 1)

if choice == '1':
    y = np.sin(X[:, 0])
    fname = "sin(x)"
elif choice == '2':
    y = np.exp(-X[:, 0]**2)
    fname = "exp(-x²)"
else:
    y = X[:, 0]**3 - X[:, 0]
    fname = "x³ - x"

y = y + np.random.randn(n) * 0.05

print(f"\nTarget: y = {fname}")

model = MOT5({
    'pop_size': 80,
    'generations': 30,
    'dim': 1,
    'max_depth': 4
})

model.fit(X, y)

exp = model.explain()
print(f"\n✅ Discovered: y = {exp['metric']}")
print(f"Complexity: {exp['complexity']}")
