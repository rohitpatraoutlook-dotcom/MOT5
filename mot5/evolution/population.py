"""
Population management for genetic algorithm
"""
import numpy as np
from symbolic.generator import ExpressionGenerator

class Individual:
    """Individual in genetic algorithm"""
    def __init__(self, tree, fitness=0.0):
        self.tree = tree
        self.fitness = fitness
        self.complexity = tree.get_complexity() if tree else 0
        self.age = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, complexity={self.complexity})"

class Population:
    def __init__(self, size, trees=None, dim=2):
        self.size = size
        self.individuals = []
        self.generator = ExpressionGenerator(dim=dim)
        
        if trees:
            for tree in trees:
                self.individuals.append(Individual(tree))
        else:
            for _ in range(size):
                tree = self.generator.random_expression()
                self.individuals.append(Individual(tree))
    
    def evaluate_all(self, X, y, fitness_func):
        for ind in self.individuals:
            try:
                ind.fitness = fitness_func(ind.tree, X, y)
                ind.complexity = ind.tree.get_complexity()
            except:
                ind.fitness = 0.0
                ind.complexity = 1000
    
    def get_best(self):
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_elite(self, n):
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_inds[:min(n, len(sorted_inds))]
    
    def get_worst(self, n):
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness)
        return sorted_inds[:min(n, len(sorted_inds))]
    
    def replace(self, new_individuals):
        self.individuals = new_individuals
        self.size = len(new_individuals)
    
    def __len__(self):
        return len(self.individuals)
    
    def __getitem__(self, idx):
        return self.individuals[idx]
