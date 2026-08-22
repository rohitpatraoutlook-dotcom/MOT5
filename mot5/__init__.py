"""
MOT5 v11.0 — Smart Equation Discovery Engine
User: from mot5 import MOT5
      model = MOT5()
      model.fit(X, y)
      print(model.explain())
"""
from .mot5_v9 import MOT5V9 as MOT5
from .maha_vault import MahaVault

__all__ = ['MOT5', 'MahaVault']
__version__ = '11.0.0'
