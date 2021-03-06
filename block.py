"""
Grady Williams
March 1, 2013

Box Sprite class, draws a box on the screen and moves then
according to a certain velocity.
"""


import pygame
import math

class Box(pygame.sprite.Sprite):
    
    def __init__(self, initial_pos, height, width):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width + 2,height])
        self.image.fill(pygame.Color('green'))
        self.rect = self.image.get_rect()
        self.rect.topleft = initial_pos
        self.left = initial_pos[0]
        self.next_update_time = 0
        self._width = width
        self._height = height
        self.need_replace = False

    def update(self,speed, current_time):
        if self.next_update_time < current_time:
            self.left += -speed
            self.rect.left = math.floor(self.left)
            self.next_update_time = current_time + 30
            if self.rect.topleft[0] <= -self._width:
                self.kill()
            
