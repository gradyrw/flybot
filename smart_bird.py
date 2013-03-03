import pygame
import numpy as np
from random import gauss
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import *
from pycuda import gpuarray
import time

class smart_bird(pygame.sprite.Sprite):
    
    def __init__(self, t_height, t_width, hPPF, gravity, tunnel_upr, tunnel_lwr, height):
        pygame.sprite.Sprite.__init__(self)
        bird = pygame.image.load('flying-bird.png')
        self.size = (t_width, t_width)
        self.image = pygame.transform.smoothscale(bird,size)
        self.rect = self.image.get_rect()
        self.rect.topleft = (1*t_wdith, height/2)
        self.vel = 0
        self.gravity = gravity
        self.controls = get_controls(t_height, t_width, hPPF, gravity, tunnel_upr, tunnel_lwr, height)

    def get_controls(t_height, t_width, hPPF, gravity, t_up, t_low, h):
        T = t_width * len(t_up) / hPPF
        K = 10000
        U_d = gpuarray.zeros(T, dtype=np.float32) + 5.0
        terminals_d = gpuarray.zeros(K, dtype = np.float32)
        states_d = gpuarray.zeros(T*K, dtpye= np.float32)
        controls_d = gpuarray.zeros(T*K, dtype = npfloat32)
        generator = XORWOWRRandomNumberGenerator()
        
        
