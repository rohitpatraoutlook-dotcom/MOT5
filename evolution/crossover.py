"""
Crossover operators for genetic algorithm
"""
import numpy as np
from symbolic.node import Node

class SubtreeCrossover:
    def __init__(self, rate=0.8):
        self.rate = rate
    
    def crossover(self, tree1, tree2):
        """Subtree swapping crossover"""
        if np.random.random() > self.rate:
            return tree1.copy(), tree2.copy()
        
        # Get all nodes from both trees
        nodes1 = tree1.get_nodes()
        nodes2 = tree2.get_nodes()
        
        # Skip root nodes sometimes
        if len(nodes1) > 2:
            nodes1 = nodes1[1:]
        if len(nodes2) > 2:
            nodes2 = nodes2[1:]
        
        # Select random nodes
        node1 = np.random.choice(nodes1)
        node2 = np.random.choice(nodes2)
        
        # Copy trees
        child1 = tree1.copy()
        child2 = tree2.copy()
        
        # Find nodes in copied trees (use value matching)
        nodes1_copy = child1.get_nodes()
        nodes2_copy = child2.get_nodes()
        
        # Match by value and structure (simplified)
        target1 = None
        target2 = None
        
        for n in nodes1_copy:
            if n.value == node1.value:
                target1 = n
                break
        
        for n in nodes2_copy:
            if n.value == node2.value:
                target2 = n
                break
        
        if target1 and target2:
            # Swap subtrees
            parent1 = target1.parent
            parent2 = target2.parent
            
            # Replace in child1
            child1 = child1.replace_subtree(target1, target2.copy())
            # Replace in child2
            child2 = child2.replace_subtree(target2, target1.copy())
        
        return child1, child2
