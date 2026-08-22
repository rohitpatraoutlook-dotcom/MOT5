"""
Ricci tensor and scalar calculator
"""
import numpy as np
from geometry.riemann import RiemannCalculator

class RicciCalculator:
    def __init__(self, metric_func):
        self.metric_func = metric_func
        self.riemann = RiemannCalculator(metric_func)
    
    def compute_tensor(self, x):
        """Compute Ricci tensor Rᵢⱼ at point x"""
        dim = len(x)
        R = self.riemann.compute(x)
        ricci = np.zeros((dim, dim))
        
        # Contract Riemann: Rᵢⱼ = Rᵏᵢⱼₖ
        for i in range(dim):
            for j in range(dim):
                ricci[i][j] = np.sum(R[:, i, j, :])
        
        return ricci
    
    def compute_scalar(self, x):
        """Compute Ricci scalar R at point x"""
        g = self.metric_func(x)
        g_inv = np.linalg.inv(g)
        ricci = self.compute_tensor(x)
        
        # R = gⁱʲ Rᵢⱼ
        R_scalar = np.sum(g_inv * ricci)
        return R_scalar
