"""
MOT5 Pro Core Engine - Shared Learning
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from mot5 import MOT5

class MOT5Pro:
    def __init__(self, n_outputs=1, batch_size=1000, config=None):
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.config = config or {}
        self.models = []
        self.shared_memory = None
        self.fitted = False
    
    def fit(self, X, y):
        """Train with shared memory across outputs"""
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        self.n_outputs = y.shape[1]
        self.models = []
        self.shared_memory = None
        
        print(f"📊 Training {self.n_outputs} output(s) with shared memory...")
        
        for i in range(self.n_outputs):
            print(f"  Output {i+1}/{self.n_outputs}")
            
            # Create model
            model = MOT5(self.config)
            
            # Share memory across models
            if self.shared_memory is not None:
                model.engine.memory = self.shared_memory
            
            # Train
            model.fit(X, y[:, i])
            
            # Update shared memory
            self.shared_memory = model.engine.memory
            self.models.append(model)
        
        self.fitted = True
        return self
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Call fit() first!")
        
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.column_stack(predictions)
    
    def explain(self, output_idx=0):
        if not self.fitted:
            return "Call fit() first!"
        if output_idx >= len(self.models):
            return f"Output {output_idx} not found"
        return self.models[output_idx].explain()
    
    def save(self, path='model_pro.json'):
        for i, model in enumerate(self.models):
            model.save(f"{path}_output_{i}")
        return f"✅ Saved {len(self.models)} models"
    
    def load(self, path='model_pro.json'):
        self.models = []
        for i in range(self.n_outputs):
            model = MOT5()
            model.load(f"{path}_output_{i}")
            self.models.append(model)
        self.fitted = True
        return f"✅ Loaded {len(self.models)} models"
