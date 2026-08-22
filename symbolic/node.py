"""
Expression Tree Node - Core symbolic representation
"""
import numpy as np
from core.constants import EPSILON
from symbolic.operators import OPERATORS

class Node:
    """Expression Tree Node"""
    
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = None
        self.depth = 0
        self.id = None
        
        if left:
            left.parent = self
        if right:
            right.parent = self
    
    def evaluate(self, X):
        """Evaluate tree on batch X (n_samples, n_features)"""
        if self.is_operator():
            left_val = self.left.evaluate(X) if self.left else None
            right_val = self.right.evaluate(X) if self.right else None
            return OPERATORS[self.value]['func'](left_val, right_val)
        elif self.is_variable():
            return X[:, self.value]
        else:
            return np.full(X.shape[0], self.value)
    
    def evaluate_single(self, x):
        """Evaluate on single point"""
        if self.is_operator():
            left_val = self.left.evaluate_single(x) if self.left else None
            right_val = self.right.evaluate_single(x) if self.right else None
            return OPERATORS[self.value]['func'](left_val, right_val)
        elif self.is_variable():
            return x[self.value]
        else:
            return self.value
    
    def is_operator(self):
        return self.value in OPERATORS
    
    def is_variable(self):
        return isinstance(self.value, int)
    
    def is_constant(self):
        return isinstance(self.value, (int, float)) and not self.is_variable()
    
    def is_zero(self):
        return self.is_constant() and abs(self.value) < EPSILON
    
    def is_one(self):
        return self.is_constant() and abs(self.value - 1.0) < EPSILON
    
    def copy(self):
        new_node = Node(self.value)
        new_node.left = self.left.copy() if self.left else None
        new_node.right = self.right.copy() if self.right else None
        new_node.depth = self.depth
        return new_node
    
    def replace_subtree(self, old_subtree, new_subtree):
        """Replace old_subtree with new_subtree in this tree"""
        if self == old_subtree:
            # This node is the one to replace
            new_subtree.parent = self.parent
            return new_subtree
        
        if self.left:
            self.left = self.left.replace_subtree(old_subtree, new_subtree)
            if self.left:
                self.left.parent = self
        
        if self.right:
            self.right = self.right.replace_subtree(old_subtree, new_subtree)
            if self.right:
                self.right.parent = self
        
        return self
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        if self.value != other.value:
            return False
        left_eq = self.left == other.left if self.left else other.left is None
        right_eq = self.right == other.right if self.right else other.right is None
        return left_eq and right_eq
    
    def to_string(self):
        if self.is_operator():
            info = OPERATORS.get(self.value, {})
            symbol = info.get('symbol', str(self.value))
            arity = info.get('arity', 2)
            if arity == 1:
                return f"{symbol}({self.left.to_string()})"
            else:
                return f"({self.left.to_string()} {symbol} {self.right.to_string()})"
        elif self.is_variable():
            return f"x{self.value}"
        else:
            return f"{self.value:.4f}"
    
    def get_complexity(self):
        count = 1
        if self.left:
            count += self.left.get_complexity()
        if self.right:
            count += self.right.get_complexity()
        return count
    
    def get_depth(self):
        if self.is_operator():
            left_depth = self.left.get_depth() if self.left else 0
            right_depth = self.right.get_depth() if self.right else 0
            return 1 + max(left_depth, right_depth)
        return 1
    
    def get_nodes(self):
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_nodes())
        if self.right:
            nodes.extend(self.right.get_nodes())
        return nodes
    
    def __repr__(self):
        return self.to_string()
