import pygame

class bird(pygame.sprite.Sprite):
    
    def __init__(self, initial_position,size):
        pygame.sprite.Sprite.__init__(self)
        bird = pygame.image.load('flying-bird.png')
        self.size = size
        self.image = pygame.transform.smoothscale(bird, size)
        self.rect = self.image.get_rect()
        self.rect.topleft = initial_position
        self.next_update_time = 0
        self.vel = 0

    def update(self, current_time,collision):
        if (self.next_update_time < current_time):
            if (collision):
                explosion = pygame.image.load('explosion.png')
                self.image = pygame.transform.smoothscale(explosion,self.size)
            self.rect.top += 0
            self.next_update_time = current_time + 25
    
