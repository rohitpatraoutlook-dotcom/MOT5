"""
Mutation operators for genetic algorithm
"""
import numpy as np
from symbolic.node import Node
from symbolic.operators import OPERATORS, UNARY_OPS, BINARY_OPS
from symbolic.generator import ExpressionGenerator

class SubtreeMutation:
    def __init__(self, rate=0.1, dim=2, max_depth=5):
        self.rate = rate
        self.generator = ExpressionGenerator(dim=dim, max_depth=max_depth)
    
    def mutate(self, tree):
        if np.random.random() > self.rate:
            return tree
        
        new_tree = tree.copy()
        
        # Get all nodes
        nodes = new_tree.get_nodes()
        
        # Select random node (skip root sometimes)
        if len(nodes) > 1 and np.random.random() < 0.3:
            node = np.random.choice(nodes[1:])
        else:
            node = np.random.choice(nodes)
        
        # Replace with random subtree
        new_subtree = self.generator.random_expression()
        
        # Replace in tree using new method
        new_tree = new_tree.replace_subtree(node, new_subtree)
        
        return new_tree

class PointMutation:
    def __init__(self, rate=0.05):
        self.rate = rate
    
    def mutate(self, tree):
        if np.random.random() > self.rate:
            return tree
        
        new_tree = tree.copy()
        nodes = new_tree.get_nodes()
        node = np.random.choice(nodes)
        
        # Randomly change operator or constant
        if node.is_operator():
            # Change operator
            ops = list(OPERATORS.keys())
            old_arity = OPERATORS[node.value]['arity']
            available = [op for op in ops if OPERATORS[op]['arity'] == old_arity]
            node.value = np.random.choice(available)
        elif node.is_constant():
            # Perturb constant
            node.value += np.random.randn() * 0.1
            if abs(node.value) < 0.001:
                node.value = np.random.randn() * 0.5
        
        return new_tree
