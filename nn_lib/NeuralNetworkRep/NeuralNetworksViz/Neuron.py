import pygame


class Neuron():

    def __init__(self, _x, _y):
        # will be needed as coordinates for position with pygame
        self.position = (_x, _y)
        self.connections = []

    def add_connection(self, c):
        self.connections.append(c)

    def display(self, screen):
        pygame.draw.circle(screen, (0, 0, 0), self.position, 20, 0)
        # Draw all its connections
        for i in range(0, len(self.connections)):
            self.connections[i].display(screen)
