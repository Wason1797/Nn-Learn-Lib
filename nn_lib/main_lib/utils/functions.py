from numba import njit, jit
import numpy as np


def translate(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2


@njit
def sigmoid(x):
    return 1/(1 + np.exp(-x))


@njit
def dsigmoid(x):
    return x * (1 - x)


@jit(forceobj=True)
def apply(arr: np.ndarray, func) -> np.ndarray:
    return np.array([[func(row_item) for row_item in row] for row in arr], dtype=np.double)
