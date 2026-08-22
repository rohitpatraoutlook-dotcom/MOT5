"""
OMNI Multiplication Route: Assemble from memory
"""
from symbolic.node import Node
from symbolic.simplifier import Simplifier

class MultiplicationRoute:
    def __init__(self, memory):
        self.memory = memory
        self.L1_global = None
        self.L2_assembled = None
        self.L3_assembled = None
    
    def assemble_L1_global(self, L1_results):
        """Aggregate L1 results into global skeleton"""
        # Simplified: take the best L1 metric
        best = max(L1_results, key=lambda x: x['fitness'])
        self.L1_global = best['metric']
        return self.L1_global
    
    def assemble_L2(self, L2_results):
        """Assemble L2: L1_global + L2 residuals"""
        self.L2_assembled = self.L1_global.copy()
        
        for result in L2_results:
            if 'residual' in result:
                # Add residual: g += δ
                self.L2_assembled = Node('add', self.L2_assembled, result['residual'])
        
        self.L2_assembled = Simplifier.simplify(self.L2_assembled)
        return self.L2_assembled
    
    def assemble_L3(self, L3_results):
        """Assemble L3: L2_assembled + L3 corrections"""
        self.L3_assembled = self.L2_assembled.copy()
        
        for result in L3_results:
            if 'correction' in result:
                self.L3_assembled = Node('add', self.L3_assembled, result['correction'])
        
        self.L3_assembled = Simplifier.simplify(self.L3_assembled)
        return self.L3_assembled
    
    def assemble_final(self):
        """Final assembly: L3_assembled"""
        return self.L3_assembled or self.L2_assembled or self.L1_global
