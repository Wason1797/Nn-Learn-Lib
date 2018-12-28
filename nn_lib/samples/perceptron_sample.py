from nn_lib.Perceptron import Perceptron as ps
from nn_lib.samples import perceptron_sample_ts as ts
from nn_lib.common import functions as fn
import pygame
from pygame.locals import *


BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

width = height = 500

pygame.init()
windowSurface = pygame.display.set_mode((width, height), 0, 32)
pygame.display.set_caption('Perceptron sample')

training_set = ts.init_set()


def run():

    neuron = ps.Perceptron(2)
    ps.random_weight_init(neuron)

    pause = None

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONUP:
                for point in training_set:
                    neuron.train([point.x, point.y], point.label, 0.1)
                    pause = None

        if pause is None:
            for point in training_set:

                mapped_x = int(fn.translate(point.x, 0, 1, 500, 0))
                mapped_y = int(fn.translate(point.y, 0, 1, 500, 0))

                if point.label == 1:
                    pygame.draw.circle(windowSurface, BLUE,
                                       (mapped_x, mapped_y), 6)
                else:
                    pygame.draw.circle(windowSurface, WHITE,
                                       (mapped_x, mapped_y), 6)

                if neuron.predict([point.x, point.y]) == point.label:
                    pygame.draw.circle(windowSurface, GREEN,
                                       (mapped_x, mapped_y), 2)
                else:
                    pygame.draw.circle(windowSurface, RED,
                                       (mapped_x, mapped_y), 2)

            pygame.draw.line(windowSurface, WHITE, (0, 0), (width, height), 2)

            pygame.display.update()
            pause = True
