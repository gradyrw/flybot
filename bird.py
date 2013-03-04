import pygame
import numpy as np
from PI2 import PI2 as brain
import pickle

class bird(pygame.sprite.Sprite):
    
    def __init__(self, mpf, gpf,vPPF, ppvm, pphm, height, width, upr, lwr):
        initial_position = (2*pphm,height/2)
        size = (10,10)
        mv_size = size[1]/ppvm*1.0
        mh_size = size[0]/pphm*1.0
        print mv_size
        print mh_size
        upr = upr[2:]
        lwr = lwr[2:]
        pygame.sprite.Sprite.__init__(self)
        bird = pygame.image.load('flying-bird.png')
        self.size = size
        self.image = pygame.transform.smoothscale(bird, size)
        self.rect = self.image.get_rect()
        self.rect.topleft = initial_position
        print self.rect.topleft
        print self.rect.bottomright
        self.next_update_time = 0
        self.vel = 0
        self.gravity = vPPF
        T = int((len(upr)-1)/mpf)
        print T
        thinker = brain(T, 10000, mpf, -gpf, mh_size, mv_size, (0,50), 1,upr,lwr,height/ppvm)
        self.controls = thinker.calc_path(.3, plot=True,max=2500)*ppvm
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

    def store(self, name):
        output = open(name, 'wb')
        pickle.dump(self.controls, output)
        output.close()

    
