"""
Save and load MOT5 model
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
from mot5 import MOT5

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X[:, 0] + 2 + np.random.randn(100) * 0.1

print("1️⃣ Training model...")
model = MOT5({'pop_size': 50, 'generations': 20, 'dim': 1})
model.fit(X, y)

exp = model.explain()
print(f"Discovered: y = {exp['metric']}")

# Save memory
print("\n2️⃣ Saving memory...")
model.save_memory('memory_vault.json')
print("✅ Saved to: memory_vault.json")

# Load in new model
print("\n3️⃣ Loading memory in new model...")
new_model = MOT5()
new_model.load_memory('memory_vault.json')
new_model.fitted = True
new_model.final_metric = new_model.engine.final_metric

# Predict with loaded model
X_test = np.array([[0.0], [1.0], [2.0]])
y_pred = new_model.predict(X_test)
print(f"Predictions: {y_pred}")
print("✅ Model loaded and predicting!")
