"""
OMNI Division Route - With Memory Writing
"""
import numpy as np
from evolution.ga import GeneticAlgorithm
from symbolic.node import Node
from symbolic.generator import ExpressionGenerator

class DivisionRoute:
    def __init__(self, memory, config=None):
        self.memory = memory
        self.config = config or {}
        self.L1_results = []
        self.L2_results = []
        self.L3_results = []
        self.gen = ExpressionGenerator(dim=2, max_depth=3)
    
    def run_L1(self, X, y, batches, verbose=True):
        """L1: 10-point batches with memory writing"""
        if verbose:
            print(f"Running L1: {len(batches)} batches × 10 points")
        
        for i, batch in enumerate(batches):
            if verbose and i % 10 == 0:
                print(f"  L1 Batch {i+1}/{len(batches)}")
            
            ga = GeneticAlgorithm({
                'pop_size': 30,
                'generations': 5,
                'dim': X.shape[1],
                'max_depth': 3
            })
            
            try:
                best = ga.run(batch['X'], batch['y'], verbose=False)
                best_tree = best.tree
                best_fitness = best.fitness
            except Exception as e:
                best_tree = self.gen.random_expression()
                best_fitness = 0.0
            
            # ✅ WRITE TO MEMORY
            try:
                # Create a simple hash
                import hashlib
                hash_key = hashlib.md5(str(best_tree).encode()).hexdigest()[:16]
                
                # Write invariant
                self.memory.write_invariant(
                    hash_key=hash_key,
                    R_expr=str(best_tree),
                    K_expr="0.0",
                    metric=best_tree,
                    topology="unknown"
                )
                
                # Write gene
                self.memory.write_gene(
                    expr=best_tree,
                    domain="general"
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️ Memory write failed: {e}")
            
            self.L1_results.append({
                'metric': best_tree,
                'fitness': best_fitness,
                'location': np.mean(batch['X'], axis=0),
                'batch_indices': batch['indices'],
                'hash': hash_key if 'hash_key' in locals() else "unknown"
            })
        
        if verbose:
            print("L1 complete!")
        return self.L1_results
    
    def run_L2(self, X, y, batches, L1_results, verbose=True):
        """L2: 100-point batches"""
        if verbose:
            print(f"Running L2: {len(batches)} batches × 100 points")
        
        for i, batch in enumerate(batches):
            if verbose and i % 10 == 0:
                print(f"  L2 Batch {i+1}/{len(batches)}")
            
            parent = self._find_parent(batch['X'], L1_results)
            base_metric = parent['metric']
            
            ga = GeneticAlgorithm({
                'pop_size': 30,
                'generations': 3,
                'dim': X.shape[1],
                'max_depth': 4
            })
            
            initial = [base_metric]
            for _ in range(10):
                variant = self._mutate_metric(base_metric)
                initial.append(variant)
            
            try:
                best = ga.run(batch['X'], batch['y'], initial_trees=initial, verbose=False)
                best_tree = best.tree
            except:
                best_tree = base_metric.copy()
            
            # ✅ WRITE TO MEMORY
            try:
                import hashlib
                hash_key = hashlib.md5(str(best_tree).encode()).hexdigest()[:16]
                
                self.memory.write_invariant(
                    hash_key=hash_key,
                    R_expr=str(best_tree),
                    K_expr="0.0",
                    metric=best_tree,
                    topology="regional"
                )
            except:
                pass
            
            self.L2_results.append({
                'metric': best_tree,
                'residual': None,
                'parent_hash': parent.get('hash', 'L1_unknown')
            })
        
        if verbose:
            print("L2 complete!")
        return self.L2_results
    
    def run_L3(self, X, y, batches, L2_results, verbose=True):
        """L3: 1000-point batches"""
        if verbose:
            print(f"Running L3: {len(batches)} batches × 1000 points")
        
        for i, batch in enumerate(batches):
            if verbose and i % 5 == 0:
                print(f"  L3 Batch {i+1}/{len(batches)}")
            
            parent = self._find_parent(batch['X'], L2_results)
            base_metric = parent['metric']
            
            ga = GeneticAlgorithm({
                'pop_size': 30,
                'generations': 3,
                'dim': X.shape[1],
                'max_depth': 4
            })
            
            initial = [base_metric]
            for _ in range(10):
                variant = self._mutate_metric(base_metric)
                initial.append(variant)
            
            try:
                best = ga.run(batch['X'], batch['y'], initial_trees=initial, verbose=False)
                best_tree = best.tree
            except:
                best_tree = base_metric.copy()
            
            # ✅ WRITE TO MEMORY
            try:
                import hashlib
                hash_key = hashlib.md5(str(best_tree).encode()).hexdigest()[:16]
                self.memory.write_invariant(
                    hash_key=hash_key,
                    R_expr=str(best_tree),
                    K_expr="0.0",
                    metric=best_tree,
                    topology="local"
                )
            except:
                pass
            
            self.L3_results.append({
                'metric': best_tree,
                'correction': None
            })
        
        if verbose:
            print("L3 complete!")
        return self.L3_results
    
    def _find_parent(self, X, results):
        if not results:
            return {'metric': Node(1.0), 'location': np.zeros(X.shape[1])}
        
        center = np.mean(X, axis=0)
        distances = []
        for r in results:
            loc = r.get('location', np.zeros_like(center))
            dist = np.linalg.norm(loc - center)
            distances.append(dist)
        
        idx = np.argmin(distances) if distances else 0
        return results[idx]
    
    def _mutate_metric(self, metric):
        return metric.copy()
