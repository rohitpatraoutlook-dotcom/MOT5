"""
Pretty printer for expression trees
"""
from symbolic.node import Node
from symbolic.operators import OPERATORS

class Printer:
    @staticmethod
    def to_string(node):
        if node is None:
            return "None"
        return node.to_string()
    
    @staticmethod
    def pretty_print(node, indent=0):
        prefix = "  " * indent
        if node is None:
            return prefix + "None\n"
        
        if node.is_operator():
            info = OPERATORS.get(node.value, {})
            symbol = info.get('symbol', node.value)
            arity = info.get('arity', 2)
            
            result = prefix + f"[{symbol}]\n"
            if node.left:
                result += Printer.pretty_print(node.left, indent + 1)
            if arity == 2 and node.right:
                result += Printer.pretty_print(node.right, indent + 1)
            return result
        
        elif node.is_variable():
            return prefix + f"x{node.value}\n"
        else:
            return prefix + f"{node.value:.4f}\n"
    
    @staticmethod
    def to_latex(node):
        if node is None:
            return ""
        if node.is_operator():
            info = OPERATORS.get(node.value, {})
            symbol = info.get('symbol', node.value)
            arity = info.get('arity', 2)
            if arity == 1:
                return f"\\{symbol}({Printer.to_latex(node.left)})"
            else:
                return f"({Printer.to_latex(node.left)} {symbol} {Printer.to_latex(node.right)})"
        elif node.is_variable():
            return f"x_{{{node.value}}}"
        else:
            return f"{node.value:.4f}"
