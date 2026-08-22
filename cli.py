"""
Command-line interface for MOT5
"""
import sys
import argparse
import numpy as np
from mot5 import MOT5

def main():
    parser = argparse.ArgumentParser(description='MOT5 OMNI Engine')
    parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    parser.add_argument('--target', type=str, help='Target column name')
    parser.add_argument('--pop', type=int, default=50, help='Population size')
    parser.add_argument('--gens', type=int, default=20, help='Generations')
    parser.add_argument('--save', type=str, help='Save memory to file')
    parser.add_argument('--load', type=str, help='Load memory from file')
    
    args = parser.parse_args()
    
    # Placeholder for actual data loading
    print("MOT5 OMNI Engine")
    print("="*50)
    
    # Example with synthetic data
    X = np.random.randn(200, 2)
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    
    model = MOT5({'pop_size': args.pop, 'generations': args.gens})
    model.fit(X, y)
    
    print("\nDiscovered metric:")
    print(model.explain())

if __name__ == '__main__':
    main()
