"""
TEST: Is MOT5 Learning & Storing?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mot5 import MOT5
import json
import time

print("="*60)
print("🧪 TEST: Learning + Storage Verification")
print("="*60)

np.random.seed(42)
X = np.random.randn(200, 1)
y = 2*X[:,0] + 1

print("1️⃣ Training MOT5...")
model = MOT5()
model.fit(X, y)

exp = model.explain()
print(f"   Discovered: {exp['metric']}")

# Force store in memory vault
print("\n2️⃣ Saving to JSON...")
model.save('memory_vault.json')
print("   ✅ Saved!")

# Wait a moment
time.sleep(0.5)

print("\n3️⃣ Checking JSON file...")
with open('memory_vault.json', 'r') as f:
    data = json.load(f)
    print(f"   Invariants: {len(data.get('invariant_store', {}))}")
    print(f"   Genes: {len(data.get('gene_library', {}))}")
    print(f"   Transitions: {len(data.get('transition_rules', {}))}")

print("\n4️⃣ Loading from JSON...")
new_model = MOT5()
new_model.load('memory_vault.json')
new_exp = new_model.explain()
print(f"   Loaded equation: {new_exp['metric']}")

print("\n5️⃣ Testing prediction...")
X_test = np.array([[0.0], [1.0], [2.0]])
y_pred = new_model.predict(X_test)
print(f"   Predictions: {y_pred}")
print(f"   Expected:    [1.0, 3.0, 5.0]")

print("\n6️⃣ Checking if memory stored anything...")
if len(data.get('invariant_store', {})) > 0:
    print("   ✅ YES! Memory Vault has data!")
else:
    print("   ❌ NO! Memory Vault is still empty!")

print("\n✅ VERDICT:", "MOT5 IS STORING!" if len(data.get('invariant_store', {})) > 0 else "MOT5 IS NOT STORING!")
