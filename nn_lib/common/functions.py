import math as mt


def translate(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2


def line(m, x, b):
    return m*x + b


def reshape_matrix_elements(_matrix, start1, stop1, start2, stop2):
    return [
        [
            translate(item, start1, stop1, start2, stop2)
            for item in row
        ]
        for row in _matrix]


def sigmoid(x):
    return 1/(1 + mt.exp(-x))


def dsigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def dsigmoid_min(x):
    return x * (1 - x)
