import nnfs
import numpy as np
import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data
X, y = spiral_data(100, classes=3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()
