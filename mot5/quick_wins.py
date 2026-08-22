"""
MOT5 Quick Wins - Immediate Improvements
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from mot5 import MOT5

class MOT5Plus(MOT5):
    """MOT5 with Quick Wins"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.progress_bar = None
        self.start_time = None
    
    def fit(self, X, y):
        """Fit with progress tracking"""
        self.start_time = time.time()
        print("📊 Training MOT5...")
        
        # Call parent fit
        result = super().fit(X, y)
        
        elapsed = time.time() - self.start_time
        print(f"✅ Training complete in {elapsed:.2f}s")
        
        return result
    
    def explain(self):
        """Explain with more details"""
        base = super().explain()
        if isinstance(base, dict):
            base['time'] = time.time() - self.start_time if self.start_time else 0
            base['version'] = '2.0.0'
        return base
    
    def save(self, path='model.json'):
        """Save with validation"""
        if self.final_metric is None:
            print("⚠️ No model to save!")
            return "❌ No model to save"
        
        # Validate metric is not constant
        metric_str = self.final_metric.to_string() if hasattr(self.final_metric, 'to_string') else str(self.final_metric)
        if metric_str.count('x') == 0 and len(metric_str) < 10:
            print("⚠️ Warning: Model seems too simple!")
        
        return super().save(path)

# Test the improved version
print("="*60)
print("🧪 TEST: MOT5+ Quick Wins")
print("="*60)

np.random.seed(42)
X = np.random.randn(200, 1)
y = 2*X[:,0] + 1

model = MOT5Plus()
model.fit(X, y)
exp = model.explain()
print(f"\n✅ Equation: {exp['metric']}")
print(f"📊 Complexity: {exp['complexity']}")

# Save
model.save('model_plus.json')

print("\n✅ MOT5+ Quick Wins Implemented!")
