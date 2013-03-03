import pygame
import numpy as np
from PI2 import PI2 as brain
class bird(pygame.sprite.Sprite):
    
    def __init__(self, mpf, gpf,vPPF, ppvm, pphm, height, width, upr, lwr):
        initial_position = (200,250)
        size = (25,25)
        mv_size = 25/ppvm*1.0
        mh_size = 25/pphm*1.0
        upr = upr[2:]
        lwr = lwr[2:]
        pygame.sprite.Sprite.__init__(self)
        bird = pygame.image.load('flying-bird.png')
        self.size = size
        self.image = pygame.transform.smoothscale(bird, size)
        self.rect = self.image.get_rect()
        self.rect.topleft = initial_position
        self.next_update_time = 0
        self.vel = np.float(0)
        self.ppvm = np.float32(ppvm)
        self.gravity = np.float32(2.5000000000000000)
        thinker = brain(400, 10000, mpf, -gpf, mh_size, mv_size, (0,50), 1,upr,lwr,100)
        self.controls = thinker.calc_path(.1, plot=True,max=2500)*self.ppvm
        self.count = 0
        self.top = initial_position[1]
        
    def update(self, current_time,collision):
        if (collision):
            explosion = pygame.image.load('explosion.png')
            self.image = pygame.transform.smoothscale(explosion, self.size)
        if (self.next_update_time < current_time and self.count < 400):
            dv = (self.gravity - self.controls[self.count])
            self.vel += dv
            self.count += 1
            self.top += self.vel
            self.rect.top = round(self.top)
            self.next_update_time = current_time + 30
    
