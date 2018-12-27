from nn_lib.Perceptron import Perceptron as ps


neuron = ps.Perceptron(2)


def run():
    print(neuron.weights)
    ps.init_random_weights(neuron)
    print(neuron.weights)
