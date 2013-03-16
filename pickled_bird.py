"""
Grady Williams
March 1, 2013

Pickled bird sprite class. This program is a copy of the normal bird program,
but it doesn't require PyCuda to run. It loads a pickled file of the controls.
This way a computer which is CUDA capable can generate the controls, dump them
to a file and then any computer can run those controls and hence the bird
animation.
"""


import pygame
import numpy as np
import pickle

class p_bird(pygame.sprite.Sprite):
    
    #Initializes the bird sprite
    def __init__(self, vPPF, pphm, height, name):
        initial_position = (2*pphm,height/2)
        size = (12,25)
        pygame.sprite.Sprite.__init__(self)
        bird = pygame.image.load('flying-bird.png')
        self.size = size
        self.image = pygame.transform.smoothscale(bird, size)
        self.rect = self.image.get_rect()
        self.rect.topleft = initial_position
        self.next_update_time = 0
        self.vel = 0
        self.gravity = vPPF
        pkl_file = open(name, 'rb')
        self.controls = pickle.load(pkl_file)
        pkl_file.close()
        self.count = 0
        self.top = initial_position[1]

    #Updates the sprites position on the screen.
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

