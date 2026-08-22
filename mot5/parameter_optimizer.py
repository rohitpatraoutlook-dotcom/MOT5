"""
Parameter Optimizer — Optimize parameters using gradient descent
"""
import numpy as np
from .operation_pool import OPERATIONS

class ParameterOptimizer:
    def __init__(self, lr=0.01, max_iter=100):
        self.lr = lr
        self.max_iter = max_iter
    
    def optimize(self, sequence, X, y):
        """
        Optimize parameters in sequence
        sequence = [('mul', 7), ('add', 1)]
        """
        # Extract parameters to optimize
        params = []
        param_indices = []
        
        for i, (op, param) in enumerate(sequence):
            if param is not None:
                params.append(float(param))
                param_indices.append(i)
        
        if not params:
            return sequence
        
        params = np.array(params)
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Forward pass
            pred = self._evaluate_sequence(sequence, X, params, param_indices)
            
            # Loss
            loss = np.mean((pred - y) ** 2)
            
            # Gradient (numerical)
            grad = self._compute_gradient(sequence, X, y, params, param_indices)
            
            # Update
            params -= self.lr * grad
        
        # Update sequence with optimized params
        for i, idx in enumerate(param_indices):
            op, _ = sequence[idx]
            sequence[idx] = (op, params[i])
        
        return sequence
    
    def _evaluate_sequence(self, sequence, X, params, param_indices):
        """Evaluate sequence on X with given parameters"""
        results = []
        for x in X.flatten():
            current = x
            param_idx = 0
            for i, (op, _) in enumerate(sequence):
                if i in param_indices:
                    param = params[param_idx]
                    param_idx += 1
                else:
                    param = None
                
                if op in OPERATIONS:
                    if OPERATIONS[op]['arity'] == 1:
                        current = OPERATIONS[op]['func'](current)
                    else:
                        current = OPERATIONS[op]['func'](current, param)
            results.append(current)
        return np.array(results)
    
    def _compute_gradient(self, sequence, X, y, params, param_indices):
        """Compute numerical gradient"""
        grad = np.zeros_like(params)
        epsilon = 1e-6
        
        for i in range(len(params)):
            # Forward
            params_plus = params.copy()
            params_plus[i] += epsilon
            pred_plus = self._evaluate_sequence(sequence, X, params_plus, param_indices)
            loss_plus = np.mean((pred_plus - y) ** 2)
            
            # Backward
            params_minus = params.copy()
            params_minus[i] -= epsilon
            pred_minus = self._evaluate_sequence(sequence, X, params_minus, param_indices)
            loss_minus = np.mean((pred_minus - y) ** 2)
            
            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return grad
