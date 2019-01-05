import pygame


class Neuron():

    def __init__(self, _x, _y, _name):
        # will be needed as coordinates for position with pygame
        self.position = (_x, _y)
        self.connections = []
        self.radius = 20
        self.sum = 0
        self.name = _name

    def add_connection(self, c):
        self.connections.append(c)

    def display(self, screen):
        pygame.draw.circle(screen, (0, 0, 0), self.position, self.radius, 0)
        if(self.radius > 20):
            self.radius -= 1
        else:
            self.radius = 20

    def fire(self):
        self.radius = 64
        for i in range(0, len(self.connections)):
            c = self.connections[i]
            c.feedforward(self.sum)

    def feedforward(self, input):
        self.sum += input
        if(self.sum > 1):
            self.fire()
            self.sum = 0
        if(self.name == "h0"):
            print(self.sum)

    def lerp(self, a, b, f):
        return (a * (1.0 - f)) + (b * f)
