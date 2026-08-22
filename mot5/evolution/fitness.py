"""
Fitness functions with proper dimension handling
"""
import numpy as np
from core.constants import EPSILON

class FitnessFunction:
    @staticmethod
    def mse(tree, X, y):
        try:
            pred = tree.evaluate(X)
            # Ensure correct dimensions
            if pred.ndim > y.ndim:
                pred = pred.squeeze()
            elif pred.ndim < y.ndim:
                pred = pred.reshape(-1)
            
            # Ensure same shape
            if pred.shape != y.shape:
                if len(pred.shape) == 1 and len(y.shape) == 1:
                    pass  # Both 1D
                else:
                    pred = pred.flatten()
                    y = y.flatten()
            
            mse = np.mean((pred - y) ** 2)
            
            # Handle NaN/Inf
            if np.isnan(mse) or np.isinf(mse):
                return 0.0
            
            return 1.0 / (1.0 + mse)
        except Exception as e:
            return 0.0
    
    @staticmethod
    def penalized_mse(tree, X, y, complexity_penalty=0.05):
        """MSE with complexity penalty"""
        try:
            pred = tree.evaluate(X)
            if pred.ndim > y.ndim:
                pred = pred.squeeze()
            elif pred.ndim < y.ndim:
                pred = pred.reshape(-1)
            
            if pred.shape != y.shape:
                pred = pred.flatten()
                y = y.flatten()
            
            mse = np.mean((pred - y) ** 2)
            complexity = tree.get_complexity()
            
            if np.isnan(mse) or np.isinf(mse):
                return 0.0
            
            return 1.0 / (1.0 + mse + complexity_penalty * complexity)
        except:
            return 0.0
