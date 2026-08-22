"""
Composition Memory — Store Combinations of Simple Functions
"""
class CompositionMemory:
    def __init__(self):
        self.compositions = {
            'sin_cos': ['sin', 'add', 'cos'],
            'sin_exp': ['sin', 'mul', 'exp'],
            'cos_exp': ['cos', 'mul', 'exp'],
            'linear_sin': ['mul', 'add', 'sin'],
        }
    
    def get(self, pattern1, pattern2):
        key = f"{pattern1}_{pattern2}"
        return self.compositions.get(key, None)
    
    def add(self, pattern1, pattern2, sequence):
        key = f"{pattern1}_{pattern2}"
        self.compositions[key] = sequence
