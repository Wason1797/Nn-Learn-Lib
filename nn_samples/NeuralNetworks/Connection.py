import pygame

class Connection:

    def __init__(self,_from,_to,_w):
        self.weight=_w
        self.a=_from
        self.b=_to
        print(_w)
    
    def display(self,screen):
        #display based in pygame, drawing the connection
        pygame.draw.lines(screen, (0,0,0), False, [self.a.position, self.b.position], int(10*self.weight))