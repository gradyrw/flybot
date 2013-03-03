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

def main(mode = 'normal', name = 'std.pkl'):
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
    tunnel_upr = np.zeros(length+2) + 80
    tunnel_lwr = np.zeros(length+2) + 20
    for x in range(length+2):
        tunnel_upr[x] = 80
        tunnel_lwr[x] = 20
    tunnel_lwr[3] = 40
    tunnel_lwr[4] = 50
    tunnel_lwr[5] = 60
    tunnel_upr[7] = 70
    tunnel_upr[8] = 60
    tunnel_upr[9] = 50
    for x in range(10,15):
        tunnel_lwr[x] = 10 + 10*(x%10)
    for x in range(15,20):
        tunnel_lwr[x] = 60
    tunnel_upr = np.require(tunnel_upr, dtype=np.float32, requirements = ['A', 'O', 'W', 'C'])
    tunnel_lwr = np.require(tunnel_lwr, dtype = np.float32, requirements = ['A','O','W','C'])
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
    main(mode = 'create', name = 'ex1.pkl')
