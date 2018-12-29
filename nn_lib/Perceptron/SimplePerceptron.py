import random as rd


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

    def __init__(self, _input_size):
        self. weights = []
        self.input_size = _input_size

    def predict(self, _inputs):
        """
            Predict

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
        return sign(w_sum)

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

        current_guess = self.predict(_inputs)
        err = label - current_guess

        for i in range(self.input_size):
            self.weights[i] += err*_inputs[i]*learning_rate


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
