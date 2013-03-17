"""
Grady Williams
gradyrw@gmail.com
March 1, 2013
flybot

Demonstration of PI^2 algorithm applied to tunnel navigation. This program takes
a tunnel as input, and uses PI^2 to determine a set of acceleration controls
to navigate the tunnel. The result of the determined set of controls is 
displayed as an animation.
"""
import pygame
import numpy as np
from pygame.locals import *
from random import randint
try:
    from bird import bird
except: ImportError
from pickled_bird import p_bird
from block import Box
import time as timer
import sys


"""
Height and width: adjust these according to your screen resolution
"""
HEIGHT = 768
WIDTH = 1366
"""
Define the horizontal veloctiy and vertical dynamics parameters

meters_per_vert/horiz_screen = the number of meters displayed on a full screen
pixels_per_horiz/vert_meter = the number of pixels per meter

mpf = meters per frame
gpf = gravity per frame
hPPF = horizontal pixels per frame for one unit of speed
vPPF = vertical pixels per frame for one unit of speed
"""
meters_per_horiz_screen = 5.0
meters_per_vert_screen = 100.0

pixels_per_horiz_meter = WIDTH/meters_per_horiz_screen
pixels_per_vert_meter = HEIGHT/meters_per_vert_screen

mpf = .05
gpf = .5
hPPF = mpf * pixels_per_horiz_meter
vPPF = gpf * pixels_per_vert_meter

"""
Defines the main loop of the tunnel navigation animation, requires a
tunnel to navigate as input. "tunnel" is a tuple of two lists, the first
element defines the upper part of the tunnel. The second defines the
lower part of the tunnel. Each tunnel list should be 23 elements long, for the
lower tunnel the elements in the list should be numbers describing how high
the tunnel is at that index, and for the higher tunnel it describes how low
the tunnel is. The overall height of the tunnel is the same as meters_per_vert_screen.

------------------
Optional Parameters
-------------------

(1) mode = 'normal', 'create', 'load

normal: creates a bird object, gets controls for the bird from PI2, and then
runs the animation if animate=True.

create: creates a bird object, gets controls for the bird from PI2, and then 
pickles the control and dumps them to a file name specified by 'name'.

load: loads controls for a given tunnel from a pickled file, then runs the animation
with those controls. Make sure the given tunnel is the same as the tunnel that
was given to create the pickled file.

(2) name = 'name'

Name of the pickle file to either dump controls to, or load controls from.

(3) animate = True/False

Tells the program whether or not to run the animation.

"""
def main(tunnel, mode = 'normal', name = 'std.pkl', animate=True):
    pygame.init()
    #Load the into CUDA readable format
    tunnel_upr, tunnel_lwr = create_tunnel(tunnel)
    boxes = pygame.sprite.RenderUpdates()
    draw_tunnel(tunnel_upr, tunnel_lwr, boxes)
    #Initialize, create and store, or load the bird
    if (mode == 'normal' or mode == 'create'):
        b = bird(mpf, gpf, vPPF,pixels_per_vert_meter, 
                 pixels_per_horiz_meter, HEIGHT, tunnel_upr,
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

"""
Loads the tunnel into a CUDA readable format
"""
def create_tunnel(tunnel):
    tunnel_upr, tunnel_lwr = tunnel
    tunnel_upr = np.require(tunnel_upr, dtype=np.float32, requirements = ['A', 'O', 'W', 'C'])
    tunnel_lwr = np.require(tunnel_lwr, dtype = np.float32, requirements = ['A','O','W','C'])
    return tunnel_upr, tunnel_lwr

"""
Draws the tunnel animation onto the screen
"""
def draw_tunnel(tunnel_upr, tunnel_lwr,boxes):
    for x in range(len(tunnel_lwr)):
        ceil = HEIGHT - tunnel_upr[x] * pixels_per_vert_meter
        floor = tunnel_lwr[x] * pixels_per_vert_meter
        ceiling_b = Box((x*pixels_per_horiz_meter, 0), int(ceil), int(pixels_per_horiz_meter))
        floor_b = Box((x*pixels_per_horiz_meter, HEIGHT - floor), int(floor),int(pixels_per_horiz_meter))
        boxes.add(floor_b)
        boxes.add(ceiling_b)

"""
Defines a hard tunnel to navigate
"""
def tunnel3():
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

"""
Defines a medium difficulty tunnel to navigate
"""
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

"""
Defines an easy tunnel to navigate
"""
def tunnel1():
    tunnel_upr = np.zeros(23)
    tunnel_lwr = np.zeros(23)
    for i in range(23):
        tunnel_lwr[i] = 10
        tunnel_upr[i] = 100 - (80*i/23.0)
    return tunnel_upr, tunnel_lwr

"""
Main runs some pre-computed demonstration by loading pickled files.
"""
if __name__ == "__main__":
    if (HEIGHT == 768):
    #Run easy, medium, and hard tunnel on a 768x1366 display
        main(tunnel1(), mode='load', name = 'tunnel1_768,1366ex.pkl')
        main(tunnel2(), mode='load', name = 'tunnel2_768,1366ex.pkl')
        main(tunnel3(), mode='load', name = 'tunnel3_768,1366ex.pkl')

    else:
    #Run easy, medium, and hard tunnel on a 1080x1920 display
        main(tunnel1(), mode='load', name = 'tunnel1_1080,1920ex.pkl')
        main(tunnel2(), mode='load', name = 'tunnel2_1080,1920ex.pkl')
        main(tunnel3(), mode='load', name = 'tunnel3_1080,1920ex.pkl')
