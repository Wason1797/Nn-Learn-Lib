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

    def train(self, _inputs, _labels, _learning_rate):

        _inputs = np.array([_inputs]).T
        _labels = np.array([_labels]).T

        # Sigmoid functions

        vsigmoid = np.vectorize(fn.sigmoid)
        vdsigmoid_min = np.vectorize(fn.dsigmoid_min)

        # Calculate the hidden layer values, input to the output layer

        hidden_layer_values = np.matmul(self.weights_input_hidden, _inputs)
        hidden_layer_values = np.add(
            hidden_layer_values, self.weights_hidden_bias)

        hidden_layer_values = vsigmoid(hidden_layer_values)

        # calculate the output from the output layer

        output_layer_values = np.matmul(
            self.weights_hidden_output, hidden_layer_values)
        output_layer_values = np.add(
            output_layer_values, self.weights_output_bias)

        output_layer_values = vsigmoid(output_layer_values)

        # Get the gradient values S'(x) for the output values matrix

        output_layer_gradient_values = vdsigmoid_min(output_layer_values)

        output_errors = np.subtract(_labels, output_layer_values)

        hidden_output_gradient = np.multiply(
            output_layer_gradient_values, output_errors)

        # Get the weight deltas

        hidden_output_delta = _learning_rate * hidden_output_gradient

        hidden_output_delta = np.matmul(
            hidden_output_delta, np.transpose(hidden_layer_values))

        # addjust weights

        self.weights_hidden_output = np.add(
            self.weights_hidden_output, hidden_output_delta)

        hidden_output_weights_transposed = np.transpose(
            self.weights_hidden_output)

        # repeat for hidden layer

        hidden_layer_gradient_values = vdsigmoid_min(hidden_layer_values)

        hidden_errors = np.matmul(
            hidden_output_weights_transposed, output_errors)

        input_hidden_gradient = np.multiply(
            hidden_layer_gradient_values, hidden_errors)

        input_hidden_delta = _learning_rate * input_hidden_gradient
        input_hidden_delta = np.matmul(
            input_hidden_delta, np.transpose(_inputs))

        self.weights_input_hidden = np.add(
            self.weights_input_hidden, input_hidden_delta)
