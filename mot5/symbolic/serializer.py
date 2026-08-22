"""
Expression Tree Serializer - Convert Tree ↔ JSON
"""
import json
from symbolic.node import Node
from symbolic.operators import OPERATORS

class TreeSerializer:
    @staticmethod
    def to_dict(node):
        """Convert Node to dictionary"""
        if node is None:
            return None
        
        result = {
            'type': 'node',
            'value': node.value,
            'is_operator': node.is_operator(),
            'is_variable': node.is_variable(),
            'is_constant': node.is_constant()
        }
        
        if node.left:
            result['left'] = TreeSerializer.to_dict(node.left)
        if node.right:
            result['right'] = TreeSerializer.to_dict(node.right)
        
        return result
    
    @staticmethod
    def from_dict(data):
        """Convert dictionary to Node"""
        if data is None:
            return None
        
        node = Node(data['value'])
        
        if 'left' in data and data['left']:
            node.left = TreeSerializer.from_dict(data['left'])
            if node.left:
                node.left.parent = node
        
        if 'right' in data and data['right']:
            node.right = TreeSerializer.from_dict(data['right'])
            if node.right:
                node.right.parent = node
        
        return node
    
    @staticmethod
    def save(tree, filepath):
        """Save tree to JSON"""
        data = TreeSerializer.to_dict(tree)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    
    @staticmethod
    def load(filepath):
        """Load tree from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return TreeSerializer.from_dict(data)
