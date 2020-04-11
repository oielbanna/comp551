from Project3.src.mlp import MLP

import numpy as np

mlp = MLP([2, 3, 2], 'relu')
print(mlp.weights[0])