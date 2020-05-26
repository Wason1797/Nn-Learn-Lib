from functools import partial

import utils.functions as fn
import random as rd
import numpy as np

import time


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        reshape = partial(fn.translate, start1=0, stop1=1, start2=-1, stop2=1)
        self.weights = {
            'i-h': fn.apply(np.random.rand(self.hidden_size, self.input_size), reshape),
            'h-o': fn.apply(np.random.rand(self.output_size, self.hidden_size), reshape),
            'h-b': fn.apply(np.random.rand(self.hidden_size, 1), reshape),
            'o-b': fn.apply(np.random.rand(self.output_size, 1), reshape),
        }

    def calculate_layer_values(self, inputs: np.ndarray, layer_key: str, bias_key: str):
        return fn.apply(np.add(np.matmul(self.weights[layer_key], inputs), self.weights[bias_key]), fn.sigmoid)

    def calculate_layer_gradient_delta(self, layer: np.ndarray, next_layer: np.ndarray, layer_errors: np.ndarray, learning_rate: float):
        layer_gradient = np.multiply(fn.apply(layer, fn.dsigmoid), layer_errors)*learning_rate
        layer_delta = np.matmul(layer_gradient, np.transpose(next_layer))
        return layer_gradient, layer_delta

    def feed_forward(self, inputs: list) -> np.ndarray:
        cu_inputs = np.array([inputs], dtype=np.double).T
        hidden_values = self.calculate_layer_values(cu_inputs, 'i-h', 'h-b')
        return self.calculate_layer_values(hidden_values, 'h-o', 'o-b')

    def train(self, inputs: list, labels: list, learning_rate: float):
        cu_inputs = np.asarray([inputs], dtype=np.double).T
        cu_labels = np.asarray([labels], dtype=np.double).T

        hidden_layer_values = self.calculate_layer_values(cu_inputs, 'i-h', 'h-b')
        output_layer_values = self.calculate_layer_values(hidden_layer_values, 'h-o', 'o-b')

        output_errors = np.subtract(cu_labels, output_layer_values)
        hidden_output_gradient, hidden_output_delta = self.calculate_layer_gradient_delta(
            output_layer_values, hidden_layer_values, output_errors, learning_rate)

        self.weights['h-o'] = np.add(self.weights['h-o'], hidden_output_delta)
        self.weights['o-b'] = np.add(self.weights['o-b'], hidden_output_gradient)

        hidden_errors = np.matmul(self.weights['h-o'].T, output_errors)

        input_hidden_gradient, input_hidden_delta = self.calculate_layer_gradient_delta(
            hidden_layer_values, cu_inputs, hidden_errors, learning_rate)

        self.weights['i-h'] = np.add(self.weights['i-h'], input_hidden_delta)
        self.weights['h-b'] = np.add(self.weights['h-b'], input_hidden_gradient)


if __name__ == "__main__":
    nn = NeuralNetwork(2, 2, 1)
    data_set = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    start = time.perf_counter()
    for _ in range(10000):
        rd.shuffle(data_set)
        for in_data, t_data in data_set:
            nn.train(in_data, t_data, 0.1)

    end = time.perf_counter()
    print("Took ->>", end-start, "s")
    print('-'*100)
    print(nn.feed_forward([0, 1]))
    print(nn.feed_forward([1, 0]))
    print(nn.feed_forward([0, 0]))
    print(nn.feed_forward([1, 1]))
    print('-'*100)
