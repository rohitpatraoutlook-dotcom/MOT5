"""
Batch sampler for OMNI multi-scale
"""
import numpy as np

class BatchSampler:
    @staticmethod
    def sample_batches(X, y, batch_size, n_batches=None):
        """Sample random batches from data"""
        n_samples = X.shape[0]
        
        if n_batches is None:
            n_batches = n_samples // batch_size
        
        batches = []
        indices = np.random.permutation(n_samples)
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            if start >= n_samples:
                break
            
            batch_idx = indices[start:end]
            batches.append({
                'X': X[batch_idx],
                'y': y[batch_idx],
                'indices': batch_idx
            })
        
        return batches
    
    @staticmethod
    def sample_balanced_batches(X, y, batch_size, n_batches=None):
        """Sample batches with balanced classes (if classification)"""
        # For regression, just use random sampling
        return BatchSampler.sample_batches(X, y, batch_size, n_batches)
