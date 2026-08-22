"""
Debug: Check if GA is working
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mot5 import MOT5
from symbolic.node import Node

np.random.seed(42)
X = np.random.randn(200, 1)
y = 2 * X[:, 0] + 1

model = MOT5({'pop_size': 50, 'generations': 10, 'dim': 1})
print("Fitting model...")
model.fit(X, y)

# Check best individual
exp = model.explain()
print(f"Equation: {exp['metric']}")
print(f"Complexity: {exp['complexity']}")

# Manually test a known good solution
test_tree = Node('add', Node('mul', Node(2.0), Node(0)), Node(1.0))
pred = test_tree.evaluate(X)
mse = np.mean((pred - y) ** 2)
print(f"Manual test (2*x + 1) MSE: {mse:.6f}")
