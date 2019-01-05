from nn_lib.NeuralNetworkRep.NeuralNetworksAnimation import Connection
from nn_lib.NeuralNetworkRep.NeuralNetworksAnimation import Neuron
from random import random
class Network:

    
    def __init__(self):
        self.neurons=[]
        self.connections=[]


    def add_neuron(self,n):
        self.neurons.append(n)


    def connect(self,a,b):
        c = Connection.Connection(a,b,random())
        a.add_connection(c)
        self.connections.append(c)


    def show(self,screen):
        #displaying the neuron based on pygame
        for i in range(0,len(self.neurons)):
            self.neurons[i].display(screen)
        for i in range(0,len(self.connections)):
            self.connections[i].display(screen)

    def feedforward(self,inputs):
        for i in range(0, len(inputs)):
            neuron=self.neurons[i]
            neuron.feedforward(inputs[i])
    
    def update(self):
        for i in range(0,len(self.connections)):
            self.connections[i].update()
    
