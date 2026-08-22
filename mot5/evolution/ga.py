"""
Genetic Algorithm - Fixed Version
"""
import numpy as np
from evolution.population import Population, Individual
from evolution.selection import TournamentSelector
from evolution.crossover import SubtreeCrossover
from evolution.mutation import SubtreeMutation
from evolution.fitness import FitnessFunction
from core.constants import EARLY_STOPPING_THRESHOLD

class GeneticAlgorithm:
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        self.pop_size = config.get('pop_size', 100)
        self.generations = config.get('generations', 30)
        self.mutation_rate = config.get('mutation_rate', 0.15)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.elite_size = config.get('elite_size', 10)
        self.tournament_size = config.get('tournament_size', 3)
        self.dim = config.get('dim', 1)
        self.max_depth = config.get('max_depth', 4)
        
        self.selector = TournamentSelector(self.tournament_size)
        self.crossover_op = SubtreeCrossover(self.crossover_rate)
        self.mutation_op = SubtreeMutation(self.mutation_rate, self.dim, self.max_depth)
        self.fitness_func = FitnessFunction.penalized_mse
        
        self.best_fitness_history = []
        self.best_individual = None
        self.population = None
    
    def run(self, X, y, initial_trees=None, verbose=True):
        """Run genetic algorithm - FIXED VERSION"""
        
        # Convert to proper shape
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) > 1:
            y = y.flatten()
        
        # Initialize population
        if initial_trees:
            self.population = Population(self.pop_size, initial_trees, self.dim)
        else:
            self.population = Population(self.pop_size, dim=self.dim)
        
        # Evaluate initial population
        self.population.evaluate_all(X, y, self.fitness_func)
        best = self.population.get_best()
        self.best_individual = best
        self.best_fitness_history.append(best.fitness)
        
        if verbose:
            print(f"Gen 0: Best fitness = {best.fitness:.4f}")
        
        # Evolution loop
        for gen in range(1, self.generations + 1):
            new_individuals = []
            
            # 1. Elitism
            elites = self.population.get_elite(self.elite_size)
            for elite in elites:
                new_individuals.append(Individual(elite.tree.copy()))
            
            # 2. Fill rest with crossover + mutation
            while len(new_individuals) < self.pop_size:
                # Select parents
                parent1 = self.selector.select(self.population)
                parent2 = self.selector.select(self.population)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1_tree, child2_tree = self.crossover_op.crossover(
                        parent1.tree, parent2.tree
                    )
                else:
                    child1_tree = parent1.tree.copy()
                    child2_tree = parent2.tree.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1_tree = self.mutation_op.mutate(child1_tree)
                if np.random.random() < self.mutation_rate:
                    child2_tree = self.mutation_op.mutate(child2_tree)
                
                new_individuals.append(Individual(child1_tree))
                if len(new_individuals) < self.pop_size:
                    new_individuals.append(Individual(child2_tree))
            
            # 3. Replace population
            self.population = Population(self.pop_size, 
                                        [ind.tree for ind in new_individuals], 
                                        self.dim)
            self.population.evaluate_all(X, y, self.fitness_func)
            
            # 4. Update best
            current_best = self.population.get_best()
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best
            
            self.best_fitness_history.append(self.best_individual.fitness)
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best fitness = {self.best_individual.fitness:.4f}")
            
            # 5. Early stopping
            if self.best_individual.fitness > EARLY_STOPPING_THRESHOLD:
                if verbose:
                    print(f"✅ Early stopping at gen {gen}")
                break
        
        return self.best_individual
