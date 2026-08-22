"""
MOT5 v4.0.0 - Equation Discovery Engine

Usage:
    from mot5 import MOT5
    model = MOT5()
    model.fit(X, y)
    print(model.explain())

For auto-sync with GitHub:
    from mot5 import MOT5Ultimate
    model = MOT5Ultimate()
    model.fit(X, y)  # Auto-syncs to GitHub!
"""

from .mot5 import MOT5
from .ultimate import MOT5Ultimate

__all__ = ['MOT5', 'MOT5Ultimate']
__version__ = '4.0.0'
