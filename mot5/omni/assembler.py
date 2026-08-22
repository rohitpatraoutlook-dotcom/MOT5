"""
Final metric assembler
"""
from symbolic.simplifier import Simplifier
from symbolic.printer import Printer

class MetricAssembler:
    @staticmethod
    def assemble(L1_global, L2_deltas, L3_corrections):
        """Assemble final metric from components"""
        # Start with L1 global
        final = L1_global.copy()
        
        # Add L2 deltas
        for delta in L2_deltas:
            final = Node('add', final, delta)
        
        # Add L3 corrections
        for corr in L3_corrections:
            final = Node('add', final, corr)
        
        # Simplify
        final = Simplifier.simplify(final)
        
        return final
    
    @staticmethod
    def to_string(metric):
        return Printer.to_string(metric)
