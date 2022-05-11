import numpy as np


def f1(X, Y):
    """
    f = sin(x) * sin(y) / (xy)
    """
    return np.sin(X) * np.sin(Y) / (X * Y)


def f2(X, Y):
    """
    a = 6.452(x + 0.125y) * [cos(x)-cos(2y)]^2
    b = [0.8 + (x-4.2)^2 + 2(y-7)^2]^1/2
    c = 3.226y
    f = a/b + c
    """
    devisor = 6.452 * (X + 0.125 * Y) * np.power(np.cos(X) - np.cos(2 * Y), 2)
    denominator = np.sqrt(0.8 + (X - 4.2) ** 2 + 2 * (Y - 7) ** 2)
    return devisor / denominator + 3.226 * Y