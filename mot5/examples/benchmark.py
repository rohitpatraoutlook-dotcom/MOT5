"""
Benchmark MOT5 performance
"""
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mot5_omni')

import numpy as np
import time
from mot5 import MOT5

def benchmark(n_samples=100, n_features=1, noise=0.1):
    """Benchmark MOT5 on different functions"""
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Test different targets
    targets = [
        ("linear", 2*X[:, 0] + 1),
        ("quadratic", X[:, 0]**2 + X[:, 0] + 1),
        ("sine", np.sin(X[:, 0])),
        ("exponential", np.exp(-X[:, 0]**2))
    ]
    
    results = []
    
    for name, y_true in targets:
        y = y_true + np.random.randn(n_samples) * noise
        
        print(f"\n🔬 Testing: {name}")
        print("-"*40)
        
        start = time.time()
        
        model = MOT5({
            'pop_size': 50,
            'generations': 15,
            'dim': n_features,
            'max_depth': 3
        })
        
        model.fit(X, y)
        
        elapsed = time.time() - start
        
        exp = model.explain()
        print(f"  Discovered: y = {exp['metric']}")
        print(f"  Complexity: {exp['complexity']}")
        print(f"  Time: {elapsed:.2f}s")
        
        # Test accuracy
        X_test = np.array([[0.0], [0.5], [1.0], [-0.5], [-1.0]])
        y_pred = model.predict(X_test)
        y_actual = [
            y_true[0] if isinstance(y_true, np.ndarray) else 0,
            # Simplified evaluation
        ]
        
        results.append({
            'name': name,
            'complexity': exp['complexity'],
            'time': elapsed
        })
    
    return results

if __name__ == '__main__':
    print("BENCHMARKING MOT5")
    print("="*50)
    results = benchmark(n_samples=100, noise=0.05)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for r in results:
        print(f"  {r['name']}: complexity={r['complexity']}, time={r['time']:.2f}s")
