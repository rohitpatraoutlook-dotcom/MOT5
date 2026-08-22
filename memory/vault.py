"""
Memory Vault - With Proper Serialization
"""
import json
import hashlib

class MemoryVault:
    def __init__(self):
        self.invariant_store = {}
        self.gene_library = {}
        self.transition_rules = {}
        self.region_maps = {}
        self.evolutionary_flow = []
        self.next_gene_id = 1
        self.best_metric = None  # Store the best metric
    
    def write_invariant(self, hash_key, R_expr, K_expr, metric, topology='unknown'):
        """Store invariant with metric"""
        if hash_key in self.invariant_store:
            self.invariant_store[hash_key]['frequency'] += 1
        else:
            # Store metric as serializable form
            metric_str = str(metric) if hasattr(metric, 'to_string') else str(metric)
            self.invariant_store[hash_key] = {
                'R_expr': str(R_expr),
                'K_expr': str(K_expr),
                'metric': metric_str,
                'metric_type': type(metric).__name__,
                'topology': topology,
                'frequency': 1,
                'trust_score': 0.5
            }
            # Store best metric
            self.best_metric = metric
        return True
    
    def get_best_metric(self):
        """Get the best metric from memory"""
        return self.best_metric
    
    def save(self, filepath):
        """Save memory to JSON"""
        # Convert metric to string for serialization
        data = {
            'invariant_store': self.invariant_store,
            'gene_library': self.gene_library,
            'transition_rules': self.transition_rules,
            'region_maps': self.region_maps,
            'evolutionary_flow': self.evolutionary_flow,
            'next_gene_id': self.next_gene_id
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    
    def load(self, filepath):
        """Load memory from JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.invariant_store = data.get('invariant_store', {})
                self.gene_library = data.get('gene_library', {})
                self.transition_rules = data.get('transition_rules', {})
                self.region_maps = data.get('region_maps', {})
                self.evolutionary_flow = data.get('evolutionary_flow', [])
                self.next_gene_id = data.get('next_gene_id', 1)
                
                # Reconstruct best metric from stored data
                if self.invariant_store:
                    for key, val in self.invariant_store.items():
                        metric_str = val.get('metric', '')
                        if metric_str:
                            # Store the metric string for later use
                            self.best_metric = metric_str
                            break
            return True
        except Exception as e:
            print(f"⚠️ Error loading memory: {e}")
            return False
