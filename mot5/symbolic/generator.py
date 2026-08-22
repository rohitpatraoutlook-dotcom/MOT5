"""
Random expression tree generator
"""
import numpy as np
from symbolic.node import Node
from symbolic.operators import OPERATORS, UNARY_OPS, BINARY_OPS

class ExpressionGenerator:
    def __init__(self, dim=2, max_depth=4, seed=None):
        self.dim = dim
        self.max_depth = max_depth
        if seed is not None:
            np.random.seed(seed)
    
    def random_expression(self, depth=0):
        """Generate random expression tree"""
        if depth >= self.max_depth or np.random.random() < 0.4:
            return self.random_leaf()
        
        # Choose operator
        if np.random.random() < 0.6:
            op = np.random.choice(BINARY_OPS)
            left = self.random_expression(depth + 1)
            right = self.random_expression(depth + 1)
            return Node(op, left, right)
        else:
            op = np.random.choice(UNARY_OPS)
            child = self.random_expression(depth + 1)
            return Node(op, child)
    
    def random_leaf(self):
        """Generate random leaf node"""
        if np.random.random() < 0.5:
            # Constant
            return Node(np.random.randn() * 2.0)
        else:
            # Variable
            return Node(np.random.randint(0, self.dim))
    
    def generate_population(self, size):
        """Generate population of random expressions"""
        return [self.random_expression() for _ in range(size)]
