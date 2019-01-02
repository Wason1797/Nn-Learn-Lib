import numpy as np
from nn_lib.common import functions as fn


class NeuralNetwork:
    def __init__(self, _in_size, _hidden_size, _out_size):
        self.in_size = _in_size
        self.hidden_size = _hidden_size
        self.out_size = _out_size
        self. weights_input_hidden = np.array(
            fn.reshape_matrix_elements(
                np.random.rand(self.hidden_size, self.in_size), 0, 1, -1, 1)
        )
        self. weights_hidden_output = np.array(
            fn.reshape_matrix_elements(
                np.random.rand(self.out_size, self.hidden_size), 0, 1, -1, 1)
        )
        self. weights_hidden_bias = np.array(
            fn.reshape_matrix_elements(
                np.random.rand(self.hidden_size, 1), 0, 1, -1, 1)
        )
        self. weights_output_bias = np.array(
            fn.reshape_matrix_elements(
                np.random.rand(self.out_size, 1), 0, 1, -1, 1)
        )

    def feed_forward_prediction(self, _inputs):
        _inputs = np.array([_inputs]).T

        vsigmoid = np.vectorize(fn.sigmoid)

        hidden_layer_values = np.matmul(self.weights_input_hidden, _inputs)
        hidden_layer_values = np.add(
            hidden_layer_values, self.weights_hidden_bias)
        hidden_layer_values = vsigmoid(hidden_layer_values)

        output_layer_values = np.matmul(
            self.weights_hidden_output, hidden_layer_values)
        output_layer_values = np.add(
            output_layer_values, self.weights_output_bias)
        output_layer_values = vsigmoid(output_layer_values)

        return output_layer_values

    def train(self, _inputs, _label):
        pass
