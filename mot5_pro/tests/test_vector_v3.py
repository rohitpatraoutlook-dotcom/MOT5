"""
TEST 1 V3: Vector Output with Diversity Preservation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from mot5 import MOT5
from evolution.ga import GeneticAlgorithm
from evolution.population import Population, Individual
from evolution.fitness import FitnessFunction

print("="*60)
print("🧪 TEST 1 V3: Diversity-Preserving GA")
print("="*60)

np.random.seed(42)
X = np.random.randn(300, 2)
y1 = 2*X[:,0] + 3*X[:,1] + 1 + np.random.randn(300) * 0.01

print(f"Target: y = 2x0 + 3x1 + 1")

# Custom GA with diversity preservation
class DiversityGA:
    def __init__(self, config=None):
        self.config = config or {}
        self.pop_size = self.config.get('pop_size', 200)
        self.generations = self.config.get('generations', 80)
        self.dim = self.config.get('dim', 2)
        self.max_depth = self.config.get('max_depth', 4)
        self.elite_size = self.config.get('elite_size', 20)
        self.mutation_rate = self.config.get('mutation_rate', 0.2)
        
        self.best = None
    
    def run(self, X, y):
        from symbolic.generator import ExpressionGenerator
        gen = ExpressionGenerator(dim=self.dim, max_depth=self.max_depth)
        
        # Initialize population
        population = []
        for _ in range(self.pop_size):
            tree = gen.random_expression()
            population.append(Individual(tree))
        
        # Evaluate
        for ind in population:
            ind.fitness = FitnessFunction.penalized_mse(ind.tree, X, y)
        
        # Track best
        self.best = max(population, key=lambda x: x.fitness)
        print(f"Gen 0: Best fitness = {self.best.fitness:.4f}")
        
        for gen in range(1, self.generations + 1):
            new_pop = []
            
            # Elite
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            new_pop.extend(sorted_pop[:self.elite_size])
            
            # Fill rest
            while len(new_pop) < self.pop_size:
                # Tournament selection
                idx1 = np.random.choice(len(population), 3, replace=False)
                idx2 = np.random.choice(len(population), 3, replace=False)
                p1 = max([population[i] for i in idx1], key=lambda x: x.fitness)
                p2 = max([population[i] for i in idx2], key=lambda x: x.fitness)
                
                # Crossover
                child1, child2 = self._crossover(p1.tree, p2.tree)
                
                # Mutation with higher rate
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, gen)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, gen)
                
                new_pop.append(Individual(child1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(Individual(child2))
            
            population = new_pop
            
            # Evaluate
            for ind in population:
                ind.fitness = FitnessFunction.penalized_mse(ind.tree, X, y)
            
            # Update best
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > self.best.fitness:
                self.best = current_best
            
            if gen % 20 == 0:
                print(f"Gen {gen}: Best fitness = {self.best.fitness:.4f}")
        
        return self.best
    
    def _crossover(self, tree1, tree2):
        from evolution.crossover import SubtreeCrossover
        return SubtreeCrossover(0.8).crossover(tree1, tree2)
    
    def _mutate(self, tree, gen):
        from evolution.mutation import SubtreeMutation
        return SubtreeMutation(0.2, self.dim, self.max_depth).mutate(tree)

# Run custom GA
ga = DiversityGA({'pop_size': 200, 'generations': 80, 'dim': 2, 'max_depth': 4})
best = ga.run(X, y1)

print(f"\n✅ Discovered: {best.tree.to_string()}")
print(f"📊 Complexity: {best.tree.get_complexity()}")
print(f"🎯 Fitness: {best.fitness:.4f}")

# Test
X_test = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_pred = best.tree.evaluate(X_test)
y_actual = 2*X_test[:,0] + 3*X_test[:,1] + 1

print(f"\n📈 Predictions:")
for i in range(4):
    print(f"  x=({X_test[i,0]}, {X_test[i,1]}) → {y_pred[i]:.2f} (expected {y_actual[i]:.2f})")

mse = np.mean((y_pred - y_actual)**2)
print(f"\n📊 MSE: {mse:.6f}")

print("\n✅ TEST 1 V3 COMPLETE!")
