import pygame
import numpy as np
from pygame.locals import *
from random import randint
from bird import bird
from block import Box
from npQueue import npQueue
import time as timer
from pycuda import gpuarray

HEIGHT = 500
WIDTH = 500
"""
Define the horizontal veloctiy and vertical dynamics parameters

meters_per_screen = the number of meters displayed on a full screen
mps = the number of meters to traverse in one second
refresh_rate = number of times (in milliseconds) the screen is refreshed

meters_per_horiz_screen
meters_per_vert_screen
mps = meters per second
gravity = -10 m/s^2

fps = frames per second

hPPM = horizontal pixels per frame
vPPM = vertical pixels per frame
"""
meters_per_horiz_screen = 10.0
meters_per_vert_screen = 10.0
mps = 1.0
gravity = 10

fps = 30

hPPF = mps * WIDTH/ (meters_per_horiz_screen * fps)
vPPF = gravity*HEIGHT/(meters_per_vert_screen * fps**2)

TILE_HEIGHT = 50
TILE_WIDTH = 50

def main():
    pygame.init()
    #Initialize the bird
    b = bird((100,250),(TILE_WIDTH,TILE_HEIGHT))
    birds = pygame.sprite.RenderUpdates()
    birds.add(b)
    #Create the initial tunnel
    tunnel_size = WIDTH/TILE_WIDTH + 3
    boxes = pygame.sprite.RenderUpdates()
    upr, lwr = draw_boxes(tunnel_size,boxes)
    num_boxes = len(boxes)
    #Intialize numpy array to keep track of tunnel information in a form
    #CUDA can read
    tunnel_upr = np.require(upr, dtype=np.float32, requirements=['A','O','W','C'])
    tunnel_lwr = np.require(lwr, dtype=np.float32, requirements=['A','O','W','C'])
    #Create the bird
    b = bird(TILE_HEIGHT, TILE_WIDTH, hPPF, -vPPF, tunnel_upr, tunnel_lrw, HEIGHT)
    #Initialize the screen to a white background
    white = [255,255,255]
    screen = pygame.display.set_mode([WIDTH,HEIGHT])
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
        boxes.update(hPPF, time)
        birds.update(time,collision,gravity)
        #Draw everythin on the screen
        birdlist = birds.draw(screen)
        rectlist = boxes.draw(screen)
        #Update the tunnel, and birds
        pygame.display.update(birdlist)
        pygame.display.update(rectlist)
        if (collision):
            birds.update(time,collision, 0)
            pygame.display.update(birds.draw(screen))
            while pygame.event.poll().type != KEYDOWN:
                True
            break
        #Clearn the screen
        birds.clear(screen, background)
        boxes.clear(screen,background)

def draw_boxes(length,boxes):
    tunnel_upr = []
    tunnel_lwr = []
    num_blocks = (WIDTH/TILE_WIDTH - 1) + 1
    ceiling_height = 0
    floor_height = 0
    ceiling_max = int(num_blocks * 3/5.0)
    floor_max = int(num_blocks * 3/5.0)
    for x in range(length):
        #Add tunnel info into tunnel keeper
        tunnel_lwr.append(floor_height*TILE_HEIGHT)
        tunnel_upr.append(HEIGHT - TILE_HEIGHT*ceiling_height)
        #Define the ceiling parameters
        ceiling_b = Box((x*TILE_WIDTH,0), TILE_HEIGHT*ceiling_height, TILE_WIDTH)
        #ceiling_height = randint(max(1,ceiling_height-2),min(ceiling_max,ceiling_height+2))
        #Define the floor parameters
        floor_b = Box((x*TILE_WIDTH,HEIGHT - TILE_HEIGHT*floor_height), TILE_HEIGHT*floor_height, TILE_WIDTH)
        #floor_height = randint(max(1, floor_height -2), min(floor_max, floor_height+2))
        #Make sure there's an opening to fly throguh
        #if (floor_height + ceiling_height > num_blocks - 8):
        #    floor_height = num_blocks - ceiling_height - 8
        ceiling_height = 2
        floor_height = 2
        boxes.add(floor_b)
        boxes.add(ceiling_b)
    return tunnel_upr, tunnel_lwr
if __name__ == "__main__":
    main()
