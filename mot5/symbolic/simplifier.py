"""
Symbolic expression simplifier
"""
import numpy as np
from symbolic.node import Node
from symbolic.operators import OPERATORS
from core.constants import EPSILON

class Simplifier:
    """Apply algebraic simplification rules"""
    
    @staticmethod
    def simplify(node):
        if node is None:
            return None
        
        # Recursively simplify children
        if node.left:
            node.left = Simplifier.simplify(node.left)
        if node.right:
            node.right = Simplifier.simplify(node.right)
        
        # Apply rules
        node = Simplifier.fold_constants(node)
        node = Simplifier.remove_identities(node)
        node = Simplifier.algebraic_rules(node)
        
        return node
    
    @staticmethod
    def fold_constants(node):
        if not node.is_operator():
            return node
        
        left_const = node.left.is_constant() if node.left else True
        right_const = node.right.is_constant() if node.right else True
        
        if left_const and right_const:
            try:
                result = node.evaluate(np.array([0.0]))
                if isinstance(result, np.ndarray):
                    result = result[0]
                return Node(float(result))
            except:
                return node
        return node
    
    @staticmethod
    def remove_identities(node):
        if not node.is_operator():
            return node
        
        op = node.value
        
        # Addition: x + 0 = x, 0 + x = x
        if op == 'add':
            if node.left.is_zero():
                return node.right
            if node.right.is_zero():
                return node.left
        
        # Subtraction: x - 0 = x, 0 - x = -x
        if op == 'sub':
            if node.right.is_zero():
                return node.left
            if node.left.is_zero():
                return Node('neg', node.right)
        
        # Multiplication: x * 1 = x, 1 * x = x, x * 0 = 0
        if op == 'mul':
            if node.left.is_zero() or node.right.is_zero():
                return Node(0.0)
            if node.left.is_one():
                return node.right
            if node.right.is_one():
                return node.left
            if node.left.value == -1:
                return Node('neg', node.right)
            if node.right.value == -1:
                return Node('neg', node.left)
        
        # Division: x / 1 = x, 0 / x = 0
        if op == 'div':
            if node.right.is_one():
                return node.left
            if node.left.is_zero():
                return Node(0.0)
        
        # Power: x^1 = x, x^0 = 1, 0^x = 0, 1^x = 1
        if op == 'pow':
            if node.right.is_zero():
                return Node(1.0)
            if node.right.is_one():
                return node.left
            if node.left.is_zero():
                return Node(0.0)
            if node.left.is_one():
                return Node(1.0)
        
        # Negation: -(-x) = x
        if op == 'neg':
            if node.left.is_constant():
                return Node(-node.left.value)
            if node.left.value == 'neg':
                return node.left.left
        
        return node
    
    @staticmethod
    def algebraic_rules(node):
        if not node.is_operator():
            return node
        
        op = node.value
        
        # x + x = 2x
        if op == 'add' and node.left == node.right:
            return Node('mul', Node(2.0), node.left)
        
        # x - x = 0
        if op == 'sub' and node.left == node.right:
            return Node(0.0)
        
        # x * x = x^2
        if op == 'mul' and node.left == node.right:
            return Node('pow', node.left, Node(2.0))
        
        # sin^2 + cos^2 = 1
        if op == 'add':
            if (node.left.value == 'pow' and node.right.value == 'pow'):
                if (node.left.left.value == 'sin' and node.right.left.value == 'cos' and
                    node.left.right.value == 2.0 and node.right.right.value == 2.0):
                    return Node(1.0)
                if (node.left.left.value == 'cos' and node.right.left.value == 'sin' and
                    node.left.right.value == 2.0 and node.right.right.value == 2.0):
                    return Node(1.0)
        
        return node
