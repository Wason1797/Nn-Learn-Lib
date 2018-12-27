import random as rd


class Perceptron:
    def __init__(self, _input_size):
        self. weights = []
        self.input_size = _input_size


def init_random_weights(_p: Perceptron):
    _p.weights.clear()
    for i in range(_p.input_size):
        _p.weights.append(rd.choice([1-rd.random(), -1+rd.random()]))
