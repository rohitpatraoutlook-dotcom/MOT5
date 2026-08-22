"""
Riemann curvature tensor calculator
"""
import numpy as np
from geometry.christoffel import ChristoffelCalculator

class RiemannCalculator:
    def __init__(self, metric_func):
        self.metric_func = metric_func
        self.christoffel = ChristoffelCalculator(metric_func)
    
    def compute(self, x):
        """Compute Riemann tensor Rᵢⱼₖˡ at point x"""
        dim = len(x)
        R = np.zeros((dim, dim, dim, dim))
        
        # Get Christoffel symbols
        gamma = self.christoffel.compute(x)
        
        # Numerical derivative of Christoffel
        epsilon = 1e-6
        dgamma = np.zeros((dim, dim, dim, dim))
        
        for k in range(dim):
            x_plus = x.copy()
            x_plus[k] += epsilon
            x_minus = x.copy()
            x_minus[k] -= epsilon
            
            gamma_plus = self.christoffel.compute(x_plus)
            gamma_minus = self.christoffel.compute(x_minus)
            dgamma[k] = (gamma_plus - gamma_minus) / (2 * epsilon)
        
        # Riemann formula
        # Rᵢⱼₖˡ = ∂Γᵢⱼˡ/∂xᵏ - ∂Γᵢₖˡ/∂xʲ + Γᵢⱼᵐ Γₘₖˡ - Γᵢₖᵐ Γₘⱼˡ
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        term1 = dgamma[k][i][j][l]
                        term2 = dgamma[j][i][k][l]
                        
                        sum3 = 0.0
                        sum4 = 0.0
                        for m in range(dim):
                            sum3 += gamma[i][j][m] * gamma[m][k][l]
                            sum4 += gamma[i][k][m] * gamma[m][j][l]
                        
                        R[i][j][k][l] = term1 - term2 + sum3 - sum4
        
        return R
