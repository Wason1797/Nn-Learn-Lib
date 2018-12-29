import random as rd
from nn_lib.common import functions as fn


class Point:
    def __init__(self, _x, _y, _label):
        self.x = _x
        self.y = _y
        self.label = _label


def init_simple_set():
    training_set = []
    ammount = 100
    for i in range(ammount):
        x = rd.random()
        y = rd.random()
        label = 1 if x > y else -1
        training_set.append(Point(x, y, label))
    return training_set


def init_refined_set():
    training_set = []
    ammount = 100
    for i in range(ammount):
        x = rd.uniform(-1, 1)
        y = rd.uniform(-1, 1)
        line_y = fn.line(0.3, x, 0.2)
        label = 1 if y > line_y else -1
        training_set.append(Point(x, y, label))
    return training_set
