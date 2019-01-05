import pygame
import sys
from nn_lib.NeuralNetworkRep.NeuralNetworksViz import Network
from pygame.locals import *

width = 1000
height = 500
Color_screen = (49, 150, 100)


def main():
    screen = pygame.display.set_mode((width, height))

    network = Network.Network()

    x0 = Network.Neuron.Neuron(-200+500, -75+200)
    x1 = Network.Neuron.Neuron(-200+500, 75+200)
    h0 = Network.Neuron.Neuron(0+500, -75+200)
    h1 = Network.Neuron.Neuron(0+500, 75+200)
    y = Network.Neuron.Neuron(200+500, 0+200)

    network.connect(x0, h0)
    network.connect(x0, h1)
    network.connect(x1, h0)
    network.connect(x1, h1)
    network.connect(h0, y)
    network.connect(h1, y)

    network.add_neuron(x0)
    network.add_neuron(x1)
    network.add_neuron(h0)
    network.add_neuron(h1)
    network.add_neuron(y)
    screen.fill(Color_screen)
    network.show(screen)
    pygame.display.flip()

    while True:
        for events in pygame.event.get():
            if events.type == QUIT:
                pygame.quit()
                return


if __name__ == '__main__':
    main()
