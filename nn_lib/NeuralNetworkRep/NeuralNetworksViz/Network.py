from nn_lib.NeuralNetworkRep.NeuralNetworksViz import Connection
from nn_lib.NeuralNetworkRep.NeuralNetworksViz import Neuron
from random import random


class Network:

    def __init__(self):
        self.neurons = []

    def add_neuron(self, n):
        self.neurons.append(n)

    def connect(self, a, b):
        c = Connection.Connection(a, b, random())
        a.add_connection(c)

    def show(self, screen):
        # displaying the neuron based on pygame
        for i in range(0, len(self.neurons)):
            self.neurons[i].display(screen)
