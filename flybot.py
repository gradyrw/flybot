import pygame
import numpy as np
from pygame.locals import *
from random import randint
from bird import bird
from pickled_bird import p_bird
from block import Box
import time as timer
import sys

HEIGHT = 1080
WIDTH = 1920
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
meters_per_horiz_screen = 5.0
meters_per_vert_screen = 100.0

pixels_per_horiz_meter = WIDTH/meters_per_horiz_screen
pixels_per_vert_meter = HEIGHT/meters_per_vert_screen

fps = 30

mpf = .05
gpf = .5
hPPF = mpf * pixels_per_horiz_meter
vPPF = gpf * pixels_per_vert_meter

def main(mode = 'normal', name = 'std.pkl', animate=True):
    pygame.init()
    #Create the tunnel
    tunnel_upr, tunnel_lwr = create_tunnel(21)
    boxes = pygame.sprite.RenderUpdates()
    draw_tunnel(tunnel_upr, tunnel_lwr, boxes)
    #Initialize, create and store, or load the bird
    if (mode == 'normal' or mode == 'create'):
        b = bird(mpf, gpf, vPPF,pixels_per_vert_meter, 
                 pixels_per_horiz_meter, HEIGHT, WIDTH, tunnel_upr,
                 tunnel_lwr)
        if (mode == 'create'):
            b.store(name)
            sys.exit()
    elif(mode == 'load'):
        b = p_bird(vPPF, pixels_per_horiz_meter, HEIGHT, name)
    if (animate==False):
        sys.exit()
    birds = pygame.sprite.RenderUpdates()
    birds.add(b)
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
        birds.update(time, collision)
        #Draw everythin on the screen
        rectlist = boxes.draw(screen)
        birdlist = birds.draw(screen)
        #Update the tunnel, and birds
        pygame.display.update(rectlist)
        pygame.display.update(birdlist)
        if (collision):
            birds.update(time,collision)
            pygame.display.update(birds.draw(screen))
            while pygame.event.poll().type != KEYDOWN:
                True
            break
        #Clearn the screen
        birds.clear(screen, background)
        boxes.clear(screen,background)

def create_tunnel(length):
    tunnel_upr, tunnel_lwr = tunnel1()
    tunnel_upr = np.require(tunnel_upr, dtype=np.float32, requirements = ['A', 'O', 'W', 'C'])
    tunnel_lwr = np.require(tunnel_lwr, dtype = np.float32, requirements = ['A','O','W','C'])
    return tunnel_upr, tunnel_lwr

def tunnel1():
    tunnel_upr = np.zeros(23)
    tunnel_lwr = np.zeros(23)
    for x in range(23):
        tunnel_upr[x] = 80
        tunnel_lwr[x] = 20
    tunnel_lwr[3] = 40
    tunnel_lwr[4] = 40
    tunnel_lwr[5] = 60
    tunnel_upr[7] = 70
    tunnel_upr[8] = 60
    tunnel_upr[9] = 50
    for x in range(10,15):
        tunnel_lwr[x] = 10 + 10*(x%10)
    tunnel_lwr[10] = 20
    for x in range(15,20):
        tunnel_lwr[x] = 60
    return tunnel_upr, tunnel_lwr

def tunnel2():
    tunnel_upr = np.zeros(23)
    tunnel_lwr = np.zeros(23)
    for x in range(3):
        tunnel_lwr[x] = 10
        tunnel_upr[x] = 90
    tunnel_lwr[3] = 30 
    tunnel_upr[3] = 70
    tunnel_lwr[4] = 30 
    tunnel_upr[4] = 60
    tunnel_lwr[5] = 20
    tunnel_upr[5] = 60
    tunnel_lwr[6] = 10
    tunnel_upr[6] = 50
    tunnel_lwr[7] = 10
    tunnel_upr[7] = 60
    tunnel_lwr[8] = 10
    tunnel_upr[8] = 70
    tunnel_lwr[9] = 30
    tunnel_upr[9] = 80
    tunnel_lwr[10] = 30
    tunnel_upr[10] = 90
    tunnel_lwr[11] = 20
    tunnel_upr[11] = 90
    tunnel_lwr[12] = 20
    tunnel_upr[12] = 90
    tunnel_lwr[13] = 30
    tunnel_upr[13] = 60
    tunnel_lwr[14] = 30
    tunnel_upr[14] = 80
    tunnel_lwr[15] = 40
    tunnel_upr[15] = 90
    tunnel_lwr[16] = 50
    tunnel_upr[16] = 90
    tunnel_lwr[17] = 60
    tunnel_upr[17] = 90
    tunnel_lwr[18] = 70
    tunnel_upr[18] = 90
    tunnel_lwr[19] = 60
    tunnel_upr[19] = 90
    tunnel_lwr[20] = 50
    tunnel_upr[20] = 85
    tunnel_lwr[21] = 40
    tunnel_upr[21] = 80
    tunnel_lwr[22] = 10
    tunnel_upr[22] = 75
    return tunnel_upr, tunnel_lwr

def tunnel3():
    tunnel_upr = np.zeros(23)
    tunnel_lwr = np.zeros(23)
    for x in range(3):
        tunnel_lwr[x] = 20
        tunnel_upr[x] = 80
    tunnel_lwr[3] = 20 
    tunnel_upr[3] = 80
    tunnel_lwr[4] = 20 
    tunnel_upr[4] = 70
    tunnel_lwr[5] = 20
    tunnel_upr[5] = 60
    tunnel_lwr[6] = 20
    tunnel_upr[6] = 50
    tunnel_lwr[7] = 20
    tunnel_upr[7] = 60
    tunnel_lwr[8] = 20
    tunnel_upr[8] = 70
    tunnel_lwr[9] = 20
    tunnel_upr[9] = 80
    tunnel_lwr[10] = 30
    tunnel_upr[10] = 80
    tunnel_lwr[11] = 40
    tunnel_upr[11] = 80
    tunnel_lwr[12] = 40
    tunnel_upr[12] = 80
    tunnel_lwr[13] = 20
    tunnel_upr[13] = 80
    tunnel_lwr[14] = 20
    tunnel_upr[14] = 70
    tunnel_lwr[15] = 20
    tunnel_upr[15] = 60
    tunnel_lwr[16] = 10
    tunnel_upr[16] = 50
    tunnel_lwr[17] = 10
    tunnel_upr[17] = 40
    tunnel_lwr[18] = 10
    tunnel_upr[18] = 50
    tunnel_lwr[19] = 10
    tunnel_upr[19] = 60
    tunnel_lwr[20] = 10
    tunnel_upr[20] = 70
    tunnel_lwr[21] = 10
    tunnel_upr[21] = 80
    tunnel_lwr[22] = 10
    tunnel_upr[22] = 80
    return tunnel_upr, tunnel_lwr
    

def draw_tunnel(tunnel_upr, tunnel_lwr,boxes):
    for x in range(len(tunnel_lwr)):
        ceil = HEIGHT - tunnel_upr[x] * pixels_per_vert_meter
        floor = tunnel_lwr[x] * pixels_per_vert_meter
        ceiling_b = Box((x*pixels_per_horiz_meter, 0), int(ceil), int(pixels_per_horiz_meter))
        floor_b = Box((x*pixels_per_horiz_meter, HEIGHT - floor), int(floor),int(pixels_per_horiz_meter))
        boxes.add(floor_b)
        boxes.add(ceiling_b)

if __name__ == "__main__":
    main(mode = 'create', name = 'demo70.pkl', animate=False)
