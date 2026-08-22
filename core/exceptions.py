"""
Custom exceptions for MOT5
"""

class MOT5Error(Exception):
    """Base exception for MOT5"""
    pass

class GeometryError(MOT5Error):
    """Raised for geometry calculation errors"""
    pass

class EvolutionError(MOT5Error):
    """Raised for evolution errors"""
    pass

class MemoryError(MOT5Error):
    """Raised for memory vault errors"""
    pass

class DataError(MOT5Error):
    """Raised for data loading errors"""
    pass

class ConvergenceError(MOT5Error):
    """Raised when evolution doesn't converge"""
    pass
