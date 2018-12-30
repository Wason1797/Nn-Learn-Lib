from nn_lib.Perceptron import RefinedPerceptron as ps
from nn_lib.samples.perceptron_samples import perceptron_sample_ts as ts
from nn_lib.common import functions as fn
import pygame
from pygame.locals import *


BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def run():

    width = height = 500

    pygame.init()
    windowSurface = pygame.display.set_mode((width, height), 0, 32)
    pygame.display.set_caption('Refined Perceptron sample')

    training_set = ts.init_refined_set()

    neuron = ps.Perceptron(2, 1)
    ps.random_weight_init(neuron)
    ps.random_bias_weight_init(neuron)
    ps.bias_value_init(neuron, 1)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONUP:
                for point in training_set:
                    neuron.train([point.x, point.y], point.label, 0.1)

        for point in training_set:

            mapped_x = int(fn.translate(point.x, -1, 1, 0, width))
            mapped_y = int(fn.translate(point.y, -1, 1, height, 0))

            if point.label == 1:
                pygame.draw.circle(windowSurface, BLUE,
                                   (mapped_x, mapped_y), 7)
            else:
                pygame.draw.circle(windowSurface, WHITE,
                                   (mapped_x, mapped_y), 7)

            if neuron.clasify([point.x, point.y]) == point.label:
                pygame.draw.circle(windowSurface, GREEN,
                                   (mapped_x, mapped_y), 4)
            else:
                pygame.draw.circle(windowSurface, RED,
                                   (mapped_x, mapped_y), 4)

        x1 = -1
        x2 = 1
        y1 = fn.line(0.3, x1, 0.2)
        y2 = fn.line(0.3, x2, 0.2)

        mapped_line1 = (fn.translate(x1, -1, 1, 0, width),
                        fn.translate(y1, -1, 1, height, 0))

        mapped_line1 = tuple(map(int, mapped_line1))

        mapped_line2 = (fn.translate(x2, -1, 1, 0, width),
                        fn.translate(y2, -1, 1, height, 0))

        mapped_line2 = tuple(map(int, mapped_line2))

        pygame.draw.line(windowSurface, WHITE,
                         mapped_line1, mapped_line2, 2)

        mapped_line1 = (fn.translate(-1, -1, 1, 0, width),
                        fn.translate(neuron.predict_line_y(-1),
                                     -1, 1, height, 0))

        mapped_line1 = tuple(map(int, mapped_line1))

        mapped_line2 = (fn.translate(1, -1, 1, 0, width),
                        fn.translate(neuron.predict_line_y(1),
                                     -1, 1, height, 0))

        mapped_line2 = tuple(map(int, mapped_line2))

        pygame.draw.line(windowSurface, RED,
                         mapped_line1, mapped_line2, 2)

        pygame.display.update()
        pygame.time.wait(100)
        windowSurface.fill(BLACK)
