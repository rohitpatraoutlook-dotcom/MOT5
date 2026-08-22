"""
Geometric invariants and hash generation
"""
import hashlib
import numpy as np
from geometry.ricci import RicciCalculator
from symbolic.node import Node

class InvariantCalculator:
    def __init__(self, metric_func):
        self.metric_func = metric_func
        self.ricci = RicciCalculator(metric_func)
    
    def compute_R_scalar(self, x):
        """Compute Ricci scalar at point x"""
        return self.ricci.compute_scalar(x)
    
    def compute_invariants(self, x):
        """Compute multiple invariants"""
        R = self.ricci.compute_scalar(x)
        # Simplified Kretschmann scalar (just for hashing)
        K = R ** 2  # Approximate
        return {'R': R, 'K': K}
    
    def generate_hash(self, x, n_samples=10):
        """Generate canonical hash from invariants"""
        # Sample multiple points
        points = []
        for _ in range(n_samples):
            # Random points in vicinity
            point = x + np.random.randn(len(x)) * 0.1
            points.append(point)
        
        # Compute invariants at each point
        R_values = []
        K_values = []
        for p in points:
            inv = self.compute_invariants(p)
            R_values.append(inv['R'])
            K_values.append(inv['K'])
        
        # Create signature string
        R_mean = np.mean(R_values)
        K_mean = np.mean(K_values)
        R_std = np.std(R_values)
        
        signature = f"R:{R_mean:.6f}:{R_std:.6f}:K:{K_mean:.6f}"
        
        # Generate hash
        hash_obj = hashlib.sha256(signature.encode())
        return hash_obj.hexdigest()[:16]
    
    @staticmethod
    def canonicalize_metric(tree):
        """Convert metric to canonical form (simplified)"""
        from symbolic.simplifier import Simplifier
        return Simplifier.simplify(tree)
