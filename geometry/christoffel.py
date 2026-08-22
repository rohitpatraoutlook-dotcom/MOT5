"""
Christoffel symbol calculator
"""
import numpy as np
from core.constants import EPSILON

class ChristoffelCalculator:
    def __init__(self, metric_func):
        """
        metric_func: function that takes (x, i, j) and returns g_ij at point x
        """
        self.metric_func = metric_func
    
    def compute(self, x):
        """Compute Christoffel symbols Γᵏᵢⱼ at point x"""
        dim = len(x)
        gamma = np.zeros((dim, dim, dim))
        
        # Compute metric at x
        g = self.metric_func(x)
        g_inv = np.linalg.inv(g)
        
        # Numerical derivative of metric
        epsilon = 1e-6
        dg = np.zeros((dim, dim, dim))
        
        for k in range(dim):
            x_plus = x.copy()
            x_plus[k] += epsilon
            x_minus = x.copy()
            x_minus[k] -= epsilon
            
            g_plus = self.metric_func(x_plus)
            g_minus = self.metric_func(x_minus)
            dg[k] = (g_plus - g_minus) / (2 * epsilon)
        
        # Christoffel formula
        # Γᵏᵢⱼ = (1/2) gᵏˡ(∂gⱼₗ/∂xⁱ + ∂gᵢₗ/∂xʲ - ∂gᵢⱼ/∂xˡ)
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    sum_terms = 0.0
                    for l in range(dim):
                        term = dg[j][l][i] + dg[i][l][j] - dg[i][j][l]
                        sum_terms += g_inv[k][l] * term
                    gamma[k][i][j] = 0.5 * sum_terms
        
        return gamma
