# MOT5 - Equation Discovery Engine

## Installation
```bash
pip install mot5
```

Usage

```python
from mot5 import MOT5
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

model = MOT5()
model.fit(X, y)
print(model.explain())
```

Features

· 3-Line API
· Auto-Sync with GitHub
· Never Forgets
· No Catastrophic Forgetting

License

MIT
