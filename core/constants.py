"""
Mathematical constants for MOT5 OMNI Engine
"""
import numpy as np

# Basic constants
PI = np.pi
E = np.e
PHI = (1 + np.sqrt(5)) / 2
GAMMA = 0.57721566490153286060

# Numerical limits
EPSILON = 1e-10
INF = 1e10
SMALL = 1e-8
LARGE = 1e6

# Algorithm defaults
DEFAULT_DIM = 2
MAX_TREE_DEPTH = 5
POPULATION_SIZE = 100
MAX_GENERATIONS = 50
ELITE_SIZE = 10
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Fitness thresholds
FITNESS_THRESHOLD = 0.99
EARLY_STOPPING_THRESHOLD = 0.95
