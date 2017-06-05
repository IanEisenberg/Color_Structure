#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:47:07 2017

@author: ian
"""

"""
ElementArray demo.
This demo requires a graphics card that supports OpenGL2 extensions.
It shows how to manipulate an arbitrary set of elements using np arrays
and avoiding for loops in your code for optimised performance.
See also the elementArrayStim demo.
"""

from psychopy import visual, event
from psychopy.visual import GratingStim
import numpy as np
from numpy import random
    
class OpticFlow(object):
    def __init__(self, win, speed, color, 
                 mask='bar', fixation_on=True, **kwargs):
        # arguments passed to ElementArray
        default_dict = {'nElements': 1000, 'sizes': .005}
        for key in default_dict:
            if key not in kwargs.keys():
                kwargs[key]=default_dict[key]
        if mask=='bar':
            mask = np.ones((100,100))
            mask[:,0:20]=0
            mask[:,80:]=0
            x = np.linspace(-np.pi, np.pi, 201)
            mask = np.vstack([np.cos(x)]*201)
        self.dots = visual.ElementArrayStim(win, elementTex=None,units = 'deg',
                                       elementMask=mask, **kwargs)
        self.base_dot_size = self.dots.sizes
        self.__dict__.update(kwargs)
        # OpticFlow specific arguments
        self.speed = speed
        self.win = win
        self.win.units = 'norm'
        # trial attributes
        self.dir = 'out'
        self.coherence = 1
        self.color = color
        # set up dots in 3d space
        # set up transformation matrices
        self.T = np.array([0,0,speed])
        # set up viewer's focal length and limits of scenece
        self.f = .5
        self.fieldlimits = [[-10,10], [-10,10], [self.f,4]] # x,y,z, min/max
        # set up dots in 3d space
        self.setupDots()
        self.project2screen()
        # set up fixation
        fix_height = .03
        ratio = win.size[1]/float(win.size[0])
        self.fixation_on = fixation_on
        self.fixation = visual.ShapeStim(self.win, 
            vertices=((0,-fix_height),(0,fix_height), 
                      (0,0), (-fix_height*ratio,0), (fix_height*ratio,0)),
            lineWidth=4,
            closeShape=False,
            lineColor="white"
        )
        
    def setupDots(self):
        self.dots3d = random.rand(self.nElements,3)
        for dim, limits in enumerate(self.fieldlimits):
            self.dots3d[:,dim]*=(limits[1]-limits[0])
            self.dots3d[:,dim]+=limits[0]

    def project2screen(self):
        projection = np.divide(self.dots3d*self.f,self.dots3d[:,2:3])[:,:2]
        # for normed units
        for dim, limits in enumerate(self.fieldlimits[0:2]):
            projection[:,dim]*=12
        self.dots.xys = projection[:,0:2]
        
    def updateTrialAttributes(self,dir=None,coherence=None,
                              color=None,speed=None,ori=None):
        if dir != None:
            assert dir in ['in','out']
            self.dir = dir
        if coherence is not None:
            assert 0 <= coherence <= 1
            self.coherence = coherence
        if color is not None:
            self.dots.setColors(color)
        if speed is not None:
            self.speed = speed
        # orientation of elements, only important for bar stim
        if ori is not None:
            self.dots.oris = ori            
        # appropriately update transformation matrix when needed
        if dir is not None or speed is not None:
            if self.dir == 'in':
                self.T[2] = -self.speed
            elif self.dir == 'out':
                self.T[2] = self.speed
        
    def updateDotsPosition(self):
        dot_coherence = np.zeros([self.nElements])
        n_coherent_dots = int((self.nElements)*self.coherence)
        dot_coherence[0:n_coherent_dots] = 1
        random.shuffle(dot_coherence)
        
        # move coherent dots
        self.dots3d[dot_coherence==1,:] -= self.T
        # move incoherent dots
        randT = random.rand((dot_coherence==0).sum(),3)-.5
        self.dots3d[dot_coherence==0,:] -= randT
        
        # replace dots that have fallen off the screen
        offscreen = self.dots3d[:,2]<self.fieldlimits[2][0]
        self.dots3d[offscreen,2] = self.fieldlimits[2][1];
        
        # replace dots that have fallen out of view
        offscreen = self.dots3d[:,2]>self.fieldlimits[2][1]
        self.dots3d[offscreen,2] = self.fieldlimits[2][0];
        
        # put points fallen off the X or Y edges back
        xlim = self.fieldlimits[0]
        ylim = self.fieldlimits[1]

        offscreen = self.dots3d[:,0:2] < [xlim[0],ylim[0]]
        adjustment = (offscreen * [xlim[1]-xlim[0], ylim[1]-ylim[0]])[offscreen]
        self.dots3d[:,0:2][offscreen] = self.dots3d[:,0:2][offscreen] + adjustment
    
        offscreen = self.dots3d[:,0:2] > [xlim[1],ylim[1]]
        adjustment = (offscreen * [xlim[1]-xlim[0], ylim[1]-ylim[0]])[offscreen]
        self.dots3d[:,0:2][offscreen] = self.dots3d[:,0:2][offscreen] - adjustment
        self.project2screen()
        
        # change dots opacities. Right at the back they should transition from 
        # to full brightness
        percent_full = np.minimum(np.abs((self.fieldlimits[2][1]-self.dots3d[:,2])/.4),1)
        self.dots.opacities = percent_full
        # change dot size
        # self.dots.sizes = np.multiply(self.base_dot_size,percent_full[:,np.newaxis])
        
    def draw(self):
        self.updateDotsPosition()
        self.dots.draw()
        if self.fixation_on:
            self.fixation.draw()
        self.win.flip()


def get_win(screen=0,fullscr=True):
    return  visual.Window([800,600], color=[-1,-1,-1], allowGUI=False, fullscr=fullscr, 
                                     monitor='testMonitor', units='norm',  screen=screen,
                                     allowStencil=True) 


"""
# For testing
win = get_win(fullscr=True)
colors = np.array([[0,0,1],[1,0,0]])
stim = OpticFlow(win,.05, color = colors[0], sizes = [.015,.02], nElements = 6000)
color_proportion = 0
while True:
    keys=event.getKeys()
    if 'q' in keys:
        break
    if keys != []:
        keys=event.waitKeys()
        
    color_proportion+=.02
    bounded_proportion = np.sin(color_proportion)*.5+.5
    color = colors[0]*bounded_proportion \
                + colors[1]*(1-bounded_proportion)
    stim.updateTrialAttributes(color = color)
    
    if color_proportion%6 > 3:
        stim.updateTrialAttributes(dir = 'in')
    else:
        stim.updateTrialAttributes(dir = 'out')
    stim.draw()
win.close()      
"""