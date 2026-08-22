"""
Geodesic equation solver (RK4)
"""
import numpy as np
from geometry.christoffel import ChristoffelCalculator

class GeodesicSolver:
    def __init__(self, metric_func):
        self.metric_func = metric_func
        self.christoffel = ChristoffelCalculator(metric_func)
    
    def solve(self, x_start, x_end, n_steps=100):
        """Solve geodesic from start to end using RK4"""
        dim = len(x_start)
        
        # Initial conditions
        direction = x_end - x_start
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return x_start
        
        # Set up ODE: state = [x, v] where v = dx/ds
        state = np.zeros(2 * dim)
        state[:dim] = x_start
        state[dim:] = direction / norm * 0.1  # Initial step
        
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            # RK4 integration
            k1 = self._geodesic_rhs(state, step * dt)
            k2 = self._geodesic_rhs(state + 0.5 * dt * k1, (step + 0.5) * dt)
            k3 = self._geodesic_rhs(state + 0.5 * dt * k2, (step + 0.5) * dt)
            k4 = self._geodesic_rhs(state + dt * k3, (step + 1) * dt)
            
            state += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return state[:dim]
    
    def _geodesic_rhs(self, state, t):
        """Right-hand side of geodesic equation"""
        dim = len(state) // 2
        x = state[:dim]
        v = state[dim:]
        
        # dx/ds = v
        dx = v
        
        # dv/ds = -Γᵏᵢⱼ vⁱ vʲ
        gamma = self.christoffel.compute(x)
        dv = np.zeros(dim)
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    dv[k] -= gamma[k][i][j] * v[i] * v[j]
        
        return np.concatenate([dx, dv])
