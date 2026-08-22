"""
Main OMNI Engine - Orchestrates everything
"""
import numpy as np
from omni.sampler import BatchSampler
from omni.division import DivisionRoute
from omni.multiplication import MultiplicationRoute
from omni.assembler import MetricAssembler
from memory.vault import MemoryVault
from symbolic.node import Node

class OMNIEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.memory = MemoryVault()
        self.division = DivisionRoute(self.memory, self.config)
        self.multiplication = MultiplicationRoute(self.memory)
        self.final_metric = Node(1.0)
        self.verbose = True
    
    def fit(self, X, y):
        """Run full OMNI pipeline"""
        if self.verbose:
            print(f"OMNI Engine: Fitting on {X.shape[0]} samples, {X.shape[1]} features")
        
        # 1. Division Route: L1, L2, L3
        if self.verbose:
            print("\n" + "="*50)
            print("PHASE 1: DIVISION ROUTE")
            print("="*50)
        
        # L1: 10-point batches
        n_batches_10 = min(20, max(1, len(X)//10))
        batches_10 = BatchSampler.sample_batches(X, y, batch_size=10, n_batches=n_batches_10)
        L1_results = self.division.run_L1(X, y, batches_10, verbose=self.verbose)
        
        # L2: 100-point batches
        n_batches_100 = min(10, max(1, len(X)//100))
        batches_100 = BatchSampler.sample_batches(X, y, batch_size=100, n_batches=n_batches_100)
        L2_results = self.division.run_L2(X, y, batches_100, L1_results, verbose=self.verbose)
        
        # L3: 1000-point batches
        n_batches_1000 = min(5, max(1, len(X)//1000))
        batches_1000 = BatchSampler.sample_batches(X, y, batch_size=1000, n_batches=n_batches_1000)
        L3_results = self.division.run_L3(X, y, batches_1000, L2_results, verbose=self.verbose)
        
        # 2. Multiplication Route: Assembly
        if self.verbose:
            print("\n" + "="*50)
            print("PHASE 2: MULTIPLICATION ROUTE")
            print("="*50)
        
        if L1_results:
            best_L1 = max(L1_results, key=lambda x: x.get('fitness', 0))
            self.final_metric = best_L1.get('metric', Node(1.0))
        else:
            self.final_metric = Node(1.0)
        
        if self.verbose:
            print("✅ OMNI complete!")
            metric_str = self.final_metric.to_string() if hasattr(self.final_metric, 'to_string') else str(self.final_metric)
            print(f"Final metric: {metric_str}")
        
        return self.final_metric
    
    def predict(self, X):
        """Predict using discovered metric"""
        if self.final_metric is None:
            raise ValueError("Model not fitted yet!")
        return self.final_metric.evaluate(X)
    
    def explain(self):
        """Get human-readable explanation"""
        if self.final_metric is None:
            return "No model fitted yet"
        
        return {
            'metric': self.final_metric.to_string() if hasattr(self.final_metric, 'to_string') else str(self.final_metric),
            'complexity': self.final_metric.get_complexity() if hasattr(self.final_metric, 'get_complexity') else 0,
            'memory_size': len(self.memory.invariant_store)
        }
