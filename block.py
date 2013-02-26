import pygame

class Box(pygame.sprite.Sprite):
    
    def __init__(self, initial_pos, height, width):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width,height])
        self.image.fill(pygame.Color('green'))
        self.rect = self.image.get_rect()
        self.rect.topleft = initial_pos
        self.next_update_time = 0
        self._width = width
        self._height = height
        self.need_replace = False

    def update(self,speed, current_time):
        if self.next_update_time < current_time:
            self.rect.left += -5
            self.next_update_time = current_time + 25
            if self.rect.topleft[0] <= -self._width:
                self.kill()
            
