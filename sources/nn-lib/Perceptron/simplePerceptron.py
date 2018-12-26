import random as rd


class Perceptron:

    def __init__(self):
        self.weigths = []
        self.size = 2

    def init_weigths(self, _weigths, _size):
        self.weigths = _weigths
        self.size = _size

    # init the weights randomly, betwen -1, 1
    def init_random(self, size):
        self.size = size
        for i in range(size):
            self.weigths.append(rd.choice([1-rd.random(), -1+rd.random()]))

    # our simple activation function
    def sign(self, n):
        return 1 if n >= 0 else -1

    def output(self, inputs):
        result = 0
        for i in range(self.size):
            result += self.weigths[i]*inputs[i]
        return self.sign(result)

    def __repr__(self):
        p_string = ""
        for w in self.weigths:
            p_string += str(w) + " "
        return p_string


p = Perceptron()
p.init_random(2)
print(p.output([-1, 0.5]))
print(p)
