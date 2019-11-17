import numpy as np

def reluDerivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def relu(x):
    x = np.maximum(x, 0)
    return x
