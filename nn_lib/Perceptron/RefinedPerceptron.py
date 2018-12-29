import random as rd
from nn_lib.common import functions as fn


class Perceptron:
    """
        The class Perceptron is the most basic representation
        of a neuron

        Atributos
        ----------
        weights : [  ]
            an array to hold the weights for the input conections
        input_size : int
            the size of the weight array, also the size of inputs expected

        Methods
        -------

        predict(_inputs: arr):
            Gives the output of the operation betwen the weights and inputs
            and conforms it to the activation function
    """

    def __init__(self, _input_size, _bias_size):
        self.weights = []
        self.input_size = _input_size
        self.biases = []
        self.bias_weights = []
        self.bias_size = _bias_size

    def clasify(self, _inputs):
        """
            Clasify

            Gives an output (weighted sum) conformed to an activation function
            in this case the sign() function
            Param
            ----------
            _inputs : arr
                The array of inputs for the perceptron
        """

        _size = len(_inputs)
        if _size != self.input_size:
            print("size missmatch {} is not equal to {}".format(
                _size, self.input_size))
            return None
        w_sum = sum([self.weights[i]*_inputs[i]
                     for i in range(self.input_size)])
        b_w_sum = sum([self.biases[i]*self.bias_weights[i]
                       for i in range(self.bias_size)])
        return sign(w_sum+b_w_sum)

    def train(self, _inputs, label, learning_rate):
        """
            Train

            Adjusts the perceptron weights acording to the error measured

            Param
            ----------
            _inputs : arr
                The array of inputs for the perceptron
            label : int
                The correct answer to those inputs
            learning_rate : float
                A variable that controlls how much we change the weights
                thowards the error
        """

        current_guess = self.clasify(_inputs)
        err = label - current_guess

        for i in range(self.input_size):
            self.weights[i] += err*_inputs[i]*learning_rate

        for i in range(self.bias_size):
            self.bias_weights[i] += err*self.biases[i]*learning_rate

    def predict_line_y(self, _x):
        if self.input_size is 2 and self.bias_size is 1:
            w0, w1 = self.weights
            w2 = self.bias_weights[0]
            return fn.line(-w0/w1, _x, -w2/w1)
        else:
            print("Non linear perceptron")
            return None


def random_weight_init(_p: Perceptron):
    """
        Random Weight Initializer

        initializes an array of weights randomly
        with numbers betwen 1 and -1

        Param
        ----------
        _p : Perceptron
            The perceptron that we want to initialize
    """

    _p.weights.clear()
    for i in range(_p.input_size):
        _p.weights.append(rd.choice([1-rd.random(), -1+rd.random()]))


def random_bias_weight_init(_p: Perceptron):
    """
        Random Bias Weight Initializer

        initializes an array of bias_weights randomly
        with numbers betwen 1 and -1

        Param
        ----------
        _p : Perceptron
            The perceptron that we want to initialize
    """

    _p.bias_weights.clear()
    for i in range(_p.bias_size):
        _p.bias_weights.append(rd.choice([1-rd.random(), -1+rd.random()]))


def bias_value_init(_p: Perceptron, _val):
    for i in range(_p.bias_size):
        _p.biases.append(_val)


def sign(num: float):
    """
        Simple Sign function

        returns 1 if the input is greater or equal to 0,
        otherwise returns -1

        side note: we will use this one as our activation function

        Param
        ----------
        num : float
            The number we want to evaluate
    """
    return 1 if num >= 0 else -1
