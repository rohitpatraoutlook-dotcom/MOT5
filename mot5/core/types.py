"""
Custom type definitions for MOT5
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Type aliases
ArrayLike = Union[np.ndarray, List[float], List[List[float]]]
TreeType = Any  # Node class
MetricType = Dict[Tuple[int, int], TreeType]  # g_ij components
TensorType = np.ndarray
HashType = str

# Data structures
class Individual:
    """Individual in genetic algorithm"""
    def __init__(self, tree, fitness: float = 0.0):
        self.tree = tree
        self.fitness = fitness
        self.complexity = 0
        self.age = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, complexity={self.complexity})"

class BatchData:
    """Batch data container"""
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: List[int]):
        self.X = X
        self.y = y
        self.indices = indices
        self.size = len(indices)
    
    def __len__(self):
        return self.size
