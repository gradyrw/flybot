"""
Grady Williams
gradyrw@gmail.com
March 1, 2013
bird

Creates a bird sprite object that navigates the tunnel, the bird calls
the PI^2 algorithm uses the set of acceleration controls it
gets to navigate a tunnel. The bird has initial position set at halfway
between the floor and ceiling. 
"""
import pygame
import numpy as np
from PI2 import PI2 as brain
import pickle
import pprint
import matplotlib.pyplot as plt
from pylab import *

"""
Initializes a sprite bird object, it takes as parameters:

mpf: meters per frame
gpf: gravity per frame
vPPF: vertical pixels per frame
ppvm: pixels per vertical meter
pphm: pixels per horizontal meter
height: the height of the tunnel
upr = upper part of the tunnel
lwr = lower part of the tunnel

The size of the object is set to 10 pixels by 10 pixels, note that
making the object too large makes it very difficult to navigate the tunnel.
"""

class bird(pygame.sprite.Sprite):   
    def __init__(self, mpf, gpf,vPPF, ppvm, pphm, height, upr, lwr):
        initial_position = (2*pphm,height/2)
        size = (15,15)
        #Defines size in terms of meters for PI^2 to use
        mv_size = size[1]/ppvm*1.0
        mh_size = size[0]/pphm*1.0
        #Cuts of the first part of the tunnel since the bird won't
        #Need to navigate it
        upr = upr[2:]
        lwr = lwr[2:]
        #Initializes the bird as a sprite object
        pygame.sprite.Sprite.__init__(self)
        #Loads the bird image, not that a different image can be used
        #by changing flying-bird.png to a different picture.
        bird = pygame.image.load('flying-bird.png')
        self.size = size
        self.image = pygame.transform.smoothscale(bird, size)
        self.rect = self.image.get_rect()
        #Puts the topleft of the bird into the initial position
        self.rect.topleft = initial_position
        self.next_update_time = 0
        self.vel = 0
        self.gravity = vPPF
        T = int((len(upr)-1)/mpf)
        #Creates an instance of the PI^2 class
        thinker = brain(T, 10000, mpf, -gpf, mv_size, mh_size, (0,50), 1,upr,lwr,height/ppvm)
        #Gets controls from PI^2
        self.controls = thinker.calc_path(.1, plot=True,max=1500)
        self.controls *= ppvm
        self.count = 0
        self.top = initial_position[1]
        
    """
    Updates the bird position and velocity, draws the bird on the screen
    """
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

    """
    Dumps the controls to a pickled file
    """
    def store(self, name):
        output = open(name, 'wb')
        pickle.dump(self.controls, output)
        output.close()

    
