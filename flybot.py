import pygame
import numpy as np
from pygame.locals import *
from random import randint
from bird import bird
from block import Box
from npQueue import npQueue
import time as timer

HEIGHT = 1080
WIDTH = 1920
"""
Define the horizontal veloctiy and vertical dynamics parameters

meters_per_screen = the number of meters displayed on a full screen
mps = the number of meters to traverse in one second
refresh_rate = number of times (in milliseconds) the screen is refreshed

PPM = pixels per meter
FPS = frames per second
PPF = pixels per frame
MPF = meters per frame
PPF = pixels per fram

NOTE: PPF needs to be an integer and evenly divide TILE_WIDTH
"""
refresh_rate = 25
FPS = 1000/refresh_rate

k = 1
meters_per_hscreen = 10.0
mps = 50.0/48

PPM = 1000/meters_per_hscreen
MPF = mps/FPS
PPF = PPM*MPF
print PPF

meters_per_vscreen = 10.0

PPM_v = 10

TILE_HEIGHT = 50
TILE_WIDTH = int(25*PPF)

def main():
    pygame.init()
    #Initialize the bird
    b = bird((100,250),(TILE_WIDTH,TILE_HEIGHT))
    birds = pygame.sprite.RenderUpdates()
    birds.add(b)
    #Create the initial tunnel
    tunnel_size = WIDTH/TILE_WIDTH + 3
    boxes = pygame.sprite.RenderUpdates()
    upr, lwr = draw_boxes(0,tunnel_size,boxes)
    num_boxes = len(boxes)
    #Intialize numpy queues to keep track of tunnel information in a form
    #Cuda can read
    tunnel_upr = npQueue(tunnel_size, upr)
    tunnel_lwr = npQueue(tunnel_size, lwr)
    #Initialize the screen to a white background
    white = [255,255,255]
    screen = pygame.display.set_mode([WIDTH,HEIGHT], FULLSCREEN)
    screen.fill(white)
    background = pygame.Surface([WIDTH,HEIGHT])
    background.fill(pygame.Color('white'))
    pygame.display.update()
    #Start the game loop, the game will continue
    #until the user presses any key
    start = 0
    while pygame.event.poll().type != KEYDOWN:
        time = pygame.time.get_ticks()
        collision = False
        #Test for a collision between the bird and blocks
        if (len(pygame.sprite.spritecollide(b,boxes,False)) > 0):
            collision = True
        #Update the locations of the boxes and bird
        boxes.update(PPF, time)
        birds.update(time,collision)
        #If we need to add onto the tunnel do so
        if (len(boxes) < num_boxes):
            new_block(tunnel_upr, tunnel_lwr, boxes, tunnel_size)
        #Draw everythin on the screen
        birdlist = birds.draw(screen)
        rectlist = boxes.draw(screen)
        #Update the tunnel, and birds
        pygame.display.update(birdlist)
        pygame.display.update(rectlist)
        #Wait 30 milliseconds to draw the next frame
        pygame.time.delay(25)
        start = timer.time()
        if (collision):
            pygame.time.delay(2500)
            break
        #Clearn the screen
        birds.clear(screen, background)
        boxes.clear(screen,background)
        #draw_boxes(tunnel_extent - 1, tunnel_extent, boxes, tunnel_keeper)

def draw_boxes(x_start, x_end, boxes):
    tunnel_upr = []
    tunnel_lwr = []
    num_blocks = HEIGHT/TILE_HEIGHT
    ceiling_height = 0
    floor_height = 0
    ceiling_max = int(num_blocks * 3/5.0)
    floor_max = int(num_blocks * 3/5.0)
    for x in range(x_start,x_end):
        #Add tunnel info into tunnel keeper
        tunnel_lwr.append(TILE_HEIGHT*floor_height)
        tunnel_upr.append(HEIGHT - TILE_HEIGHT*ceiling_height)
        #Define the ceiling parameters
        ceiling_b = Box((x*TILE_WIDTH,0), TILE_HEIGHT*ceiling_height, TILE_WIDTH)
        ceiling_height = randint(max(1,ceiling_height-2),min(ceiling_max,ceiling_height+2))
        #Define the floor parameters
        floor_b = Box((x*TILE_WIDTH,HEIGHT - TILE_HEIGHT*floor_height), TILE_HEIGHT*floor_height, TILE_WIDTH)
        floor_height = randint(max(1, floor_height -2), min(floor_max, floor_height+2))
        #Make sure there's an opening to fly throguh
        if (floor_height + ceiling_height > num_blocks - 8):
            floor_height = num_blocks - ceiling_height - 8
        boxes.add(floor_b)
        boxes.add(ceiling_b)
    return tunnel_upr, tunnel_lwr

def new_block(tunnel_upr, tunnel_lwr, boxes, tunnel_size):
    x = tunnel_size-1
    floor_height = tunnel_lwr.data[tunnel_size - 1]/TILE_HEIGHT
    ceiling_height = -(tunnel_upr.data[tunnel_size - 1] - HEIGHT)/TILE_HEIGHT
    num_blocks = HEIGHT/TILE_HEIGHT
    floor_max = int(num_blocks*3/4.0)
    ceiling_max = int(num_blocks*3/4.0)
    #Define the ceiling parameters
    ceiling_height = randint(max(1,ceiling_height-2),min(ceiling_max,ceiling_height+2))
    ceiling_b = Box((x*TILE_WIDTH,0), TILE_HEIGHT*ceiling_height, TILE_WIDTH)
    #Define the floor parameters
    floor_height = randint(max(1, floor_height -2), min(floor_max, floor_height+2))
    #Make sure there's an opening to fly throguh
    if (HEIGHT/TILE_HEIGHT - ceiling_height - floor_height < 6):
        if (floor_height > ceiling_height):
            floor_height += -6
        else:
            ceiling_height += -6
    floor_b = Box((x*TILE_WIDTH,HEIGHT - TILE_HEIGHT*floor_height), TILE_HEIGHT*floor_height, TILE_WIDTH)
    boxes.add(floor_b)
    boxes.add(ceiling_b)
    tunnel_upr.pop_add(HEIGHT - TILE_HEIGHT*ceiling_height)
    tunnel_lwr.pop_add(floor_height * TILE_HEIGHT)

if __name__ == "__main__":
    main()
