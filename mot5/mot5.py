"""
MOT5 - Ultimate Fix: Complete Model Serialization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from omni.engine import OMNIEngine
from memory.vault import MemoryVault
from symbolic.node import Node
from symbolic.serializer import TreeSerializer

class MOT5:
    def __init__(self, config=None):
        self.config = config or {}
        self.engine = OMNIEngine(self.config)
        self.fitted = False
        self.verbose = False
        self.final_metric = None
    
    def fit(self, X, y):
        X = self._prepare(X)
        y = self._prepare(y)
        self.engine.verbose = self.verbose
        self.final_metric = self.engine.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Call fit() first!")
        X = self._prepare(X)
        if self.final_metric is None:
            raise ValueError("No model loaded!")
        return self.final_metric.evaluate(X)
    
    def explain(self):
        if not self.fitted:
            return "Call fit() first!"
        if self.final_metric is None:
            return "No model loaded!"
        return {
            'metric': self.final_metric.to_string() if hasattr(self.final_metric, 'to_string') else str(self.final_metric),
            'complexity': self.final_metric.get_complexity() if hasattr(self.final_metric, 'get_complexity') else 0,
            'memory_size': len(self.engine.memory.invariant_store)
        }
    
    def save(self, path='model.json'):
        """Save full model state"""
        # 1. Save memory vault
        self.engine.memory.save(path)
        
        # 2. Save the metric tree
        tree_path = path.replace('.json', '_tree.json')
        if self.final_metric:
            TreeSerializer.save(self.final_metric, tree_path)
        
        # 3. Save metadata
        meta_path = path.replace('.json', '_meta.json')
        meta_data = {
            'fitted': self.fitted,
            'tree_file': tree_path,
            'memory_file': path,
            'version': '2.0.0'
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        return f"✅ Saved model to {path}, {tree_path}, {meta_path}"
    
    def load(self, path='model.json'):
        """Load full model state"""
        # 1. Load metadata
        meta_path = path.replace('.json', '_meta.json')
        try:
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
                self.fitted = meta_data.get('fitted', False)
                tree_path = meta_data.get('tree_file', path.replace('.json', '_tree.json'))
        except:
            tree_path = path.replace('.json', '_tree.json')
        
        # 2. Load memory vault
        self.engine.memory.load(path)
        
        # 3. Load the metric tree
        try:
            self.final_metric = TreeSerializer.load(tree_path)
            self.fitted = True
        except Exception as e:
            print(f"⚠️ Could not load tree: {e}")
            self.final_metric = Node(1.0)
            self.fitted = True
        
        return f"✅ Loaded model from {path}"
    
    def _prepare(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return data

__all__ = ['MOT5']
