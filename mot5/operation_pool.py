"""
Operation Pool — Hardcoded Operations Only
Koi numbers store nahi honge!
"""
import numpy as np

OPERATIONS = {
    'add': {'func': lambda a, b: a + b, 'arity': 2, 'symbol': '+'},
    'sub': {'func': lambda a, b: a - b, 'arity': 2, 'symbol': '-'},
    'mul': {'func': lambda a, b: a * b, 'arity': 2, 'symbol': '*'},
    'div': {'func': lambda a, b: a / b if b != 0 else 1e10, 'arity': 2, 'symbol': '/'},
    'sin': {'func': np.sin, 'arity': 1, 'symbol': 'sin'},
    'cos': {'func': np.cos, 'arity': 1, 'symbol': 'cos'},
    'tan': {'func': np.tan, 'arity': 1, 'symbol': 'tan'},
    'exp': {'func': np.exp, 'arity': 1, 'symbol': 'exp'},
    'log': {'func': lambda x: np.log(np.abs(x) + 1e-10), 'arity': 1, 'symbol': 'log'},
    'sqrt': {'func': lambda x: np.sqrt(np.abs(x)), 'arity': 1, 'symbol': 'sqrt'},
    'pow': {'func': lambda a, b: np.power(np.abs(a) + 1e-10, b), 'arity': 2, 'symbol': '^'},
    'abs': {'func': np.abs, 'arity': 1, 'symbol': 'abs'},
    'neg': {'func': lambda x: -x, 'arity': 1, 'symbol': 'neg'},
}

OPERATION_NAMES = list(OPERATIONS.keys())
