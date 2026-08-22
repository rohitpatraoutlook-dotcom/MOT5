"""
Selection operators for genetic algorithm
"""
import numpy as np

class TournamentSelector:
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size
    
    def select(self, population):
        """Tournament selection"""
        idx = np.random.choice(len(population), self.tournament_size, replace=False)
        best_idx = max(idx, key=lambda i: population[i].fitness)
        return population[best_idx]

class RouletteSelector:
    def __init__(self):
        pass
    
    def select(self, population):
        """Roulette wheel selection"""
        fitnesses = np.array([ind.fitness for ind in population])
        # Shift to positive
        fitnesses = fitnesses - np.min(fitnesses) + 1e-10
        probs = fitnesses / np.sum(fitnesses)
        idx = np.random.choice(len(population), p=probs)
        return population[idx]
