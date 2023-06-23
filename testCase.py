import numpy as np


def brusselator(a, b):
    def f(Y, t=0):
        x, y = Y
        return np.array([a + x ** 2 * y - (b + 1) * x, b * x - x ** 2 * y])
    return f