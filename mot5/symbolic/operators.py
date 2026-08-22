"""
All mathematical operators registry
"""
import numpy as np
from core.constants import EPSILON

# Safe functions
def safe_div(a, b):
    return np.divide(a, b, where=np.abs(b)>EPSILON, out=np.full_like(a, 1e10))

def safe_log(a):
    return np.log(np.abs(a) + EPSILON)

def safe_sqrt(a):
    return np.sqrt(np.abs(a))

def safe_pow(a, b):
    return np.power(np.abs(a) + EPSILON, b)

def safe_tanh(a):
    """Safe tanh with clipping"""
    return np.tanh(np.clip(a, -50, 50))

# Operator registry
OPERATORS = {
    # Binary arithmetic
    'add': {
        'func': lambda a, b: a + b if b is not None else a,
        'arity': 2, 'symbol': '+', 'complexity': 1,
        'commutative': True, 'associative': True
    },
    'sub': {
        'func': lambda a, b: a - b if b is not None else a,
        'arity': 2, 'symbol': '-', 'complexity': 1,
        'commutative': False, 'associative': False
    },
    'mul': {
        'func': lambda a, b: a * b if b is not None else a,
        'arity': 2, 'symbol': '*', 'complexity': 1,
        'commutative': True, 'associative': True
    },
    'div': {
        'func': safe_div,
        'arity': 2, 'symbol': '/', 'complexity': 2,
        'commutative': False, 'associative': False
    },
    'pow': {
        'func': safe_pow,
        'arity': 2, 'symbol': '^', 'complexity': 3,
        'commutative': False, 'associative': False
    },
    
    # Unary trig
    'sin': {
        'func': lambda a, b: np.sin(a),
        'arity': 1, 'symbol': 'sin', 'complexity': 2
    },
    'cos': {
        'func': lambda a, b: np.cos(a),
        'arity': 1, 'symbol': 'cos', 'complexity': 2
    },
    'tan': {
        'func': lambda a, b: np.tan(a),
        'arity': 1, 'symbol': 'tan', 'complexity': 2
    },
    
    # Unary inverse trig
    'asin': {
        'func': lambda a, b: np.arcsin(np.clip(a, -1, 1)),
        'arity': 1, 'symbol': 'asin', 'complexity': 3
    },
    'acos': {
        'func': lambda a, b: np.arccos(np.clip(a, -1, 1)),
        'arity': 1, 'symbol': 'acos', 'complexity': 3
    },
    'atan': {
        'func': lambda a, b: np.arctan(a),
        'arity': 1, 'symbol': 'atan', 'complexity': 3
    },
    
    # Unary exp/log
    'exp': {
        'func': lambda a, b: np.exp(np.clip(a, -50, 50)),
        'arity': 1, 'symbol': 'exp', 'complexity': 2
    },
    'log': {
        'func': safe_log,
        'arity': 1, 'symbol': 'log', 'complexity': 3
    },
    'sqrt': {
        'func': safe_sqrt,
        'arity': 1, 'symbol': 'sqrt', 'complexity': 2
    },
    
    # Unary hyperbolic (with safe clipping)
    'sinh': {
        'func': lambda a, b: np.sinh(np.clip(a, -50, 50)),
        'arity': 1, 'symbol': 'sinh', 'complexity': 3
    },
    'cosh': {
        'func': lambda a, b: np.cosh(np.clip(a, -50, 50)),
        'arity': 1, 'symbol': 'cosh', 'complexity': 3
    },
    'tanh': {
        'func': lambda a, b: safe_tanh(a),
        'arity': 1, 'symbol': 'tanh', 'complexity': 3
    },
    
    # Special
    'abs': {
        'func': lambda a, b: np.abs(a),
        'arity': 1, 'symbol': 'abs', 'complexity': 1
    },
    'neg': {
        'func': lambda a, b: -a,
        'arity': 1, 'symbol': 'neg', 'complexity': 1
    }
}

# Operator names
OPERATOR_NAMES = list(OPERATORS.keys())
UNARY_OPS = [op for op, info in OPERATORS.items() if info['arity'] == 1]
BINARY_OPS = [op for op, info in OPERATORS.items() if info['arity'] == 2]
