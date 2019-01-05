import pygame
import copy
import math

class Connection:

    def __init__(self,_from,_to,_w):
        self.weight=_w
        self.a=_from
        self.b=_to
        self.sending=False
        self.sender=()
        self.output=0
    
    def display(self,screen):
        #display based in pygame, drawing the connection
        pygame.draw.lines(screen, (0,0,0), False, [self.a.position, self.b.position], int(self.weight*10))
        if(self.sending):
            pygame.draw.circle(screen,(0,0,0),self.sender,16, 0)


    def feedforward(self,val):
        self.output=val*self.weight
        self.sender= copy.copy(self.a.position)
        self.sending=True
    
    def update(self):
        if(self.sending):
            aux_list=list(self.sender)
            aux_list[0]=int(self.lerp(self.sender[0],self.b.position[0],0.1))
            aux_list[1]=int(self.lerp(self.sender[1],self.b.position[1],0.1))
            self.sender=tuple(aux_list)
            distance=math.sqrt((self.sender[0] - self.b.position[0])**2 + (self.sender[1] - self.b.position[1])**2)
            print(distance)
            if(distance<15):
                self.b.feedforward(self.output)
                self.sending=False

    
    def lerp(self,a,b,f):
        return (a * (1.0 - f)) + (b * f)
