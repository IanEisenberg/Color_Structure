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

from psychopy import visual, event, core
from psychopy.tools.coordinatetools import pol2cart
import numpy as np
from numpy import random
   
class OpticFlow(object):
    def __init__(self, win, speed, color, **kwargs):
        # arguments passed to ElementArray
        default_dict = {'nElements': 1000, 'sizes': .005}
        for key in default_dict:
            if key not in kwargs.keys():
                kwargs[key]=default_dict[key]
        self.dots = visual.ElementArrayStim(win, elementTex=None, 
                                       elementMask='circle', **kwargs)
        self.__dict__.update(kwargs)
        # OpticFlow specific arguments
        self.speed = speed
        self.win = win
        self.win.units = 'norm'
        # trial attributes
        self.dir = 'out'
        self.coherence = .5
        self.color = color
        # set up dots in 3d space
        # set up transformation matrices
        self.T = np.array([0,0,speed])
        self.R = np.array([0,0,0])
        # set up viewer's focal length and limits of scenece
        self.f = .5
        self.fieldlimits = [[-10,10], [-10,10], [self.f,10]] # x,y,z, min/max
        # set up dots in 3d space
        self.dots3d = random.rand(self.nElements,3)
        for dim, limits in enumerate(self.fieldlimits):
            self.dots3d[:,dim]*=(limits[1]-limits[0])
            self.dots3d[:,dim]+=limits[0]
        self.project2screen()
        
      
    def project2screen(self):
        projection = np.divide(self.dots3d*self.f,self.dots3d[:,2:3])
        # for normed units
        for dim, limits in enumerate(self.fieldlimits):
            projection[:,dim]/=(np.abs(limits).sum()/2)
        aspect_ratio = float(self.win.size[1])/self.win.size[0]  # to account for wider aspect ratio
        projection[:,0]*=aspect_ratio
        self.dots.xys = projection[:,0:2]
        
    def updateTrialAttributes(self,dir=None,coherence=None,color=None):
        if dir != None:
            assert dir in ['in','out']
            self.dir = dir
            if dir == 'in':
                self.T[2] = -self.speed
            elif dir == 'out':
                self.T[2] = self.speed
        if coherence != None:
            assert 0 <= coherence <= 1
            self.coherence = coherence
        if color != None:
            self.dots.setColors(color)
        
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
        
    def draw(self):
        self.updateDotsPosition()
        self.dots.draw()
        self.win.flip()


def get_win(screen=0,fullscr=True):
    return  visual.Window([1024,768], color=[-1,-1,-1], allowGUI=False, fullscr=fullscr, 
                                     monitor='testMonitor', units='norm',  screen=screen) 

"""
win = get_win(fullscr=True)
colors = np.array([[0,0,1],[1,0,0]])
stim = OpticFlow(win,.1, color = colors[0], sizes = .008, nElements = 6000)
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

"""

class RadialOpticFlow(object):
    def __init__(self, win, maxSpeed, baseColors, radius, **kwargs):
        default_dict = {'nElements': 1000, 'sizes': .005}
        for key in default_dict:
            if key not in kwargs.keys():
                kwargs[key]=default_dict[key]
        self.dots = visual.ElementArrayStim(win, elementTex=None, 
                                       elementMask='circle', **kwargs)
        self.__dict__.update(kwargs)
        self.maxSpeed = maxSpeed
        self.radius = radius
        self.baseColors = np.array(baseColors)
        self.dotsTheta, self.dotsRadius, self.speed = self.getDotAttributes()
        self.win = win
        # trial attributes
        self.dir = 'out'
        self.coherence = .7
        self.color_proportion = 1
        
    def getDotAttributes(self):
        nDots = self.nElements
        dotsTheta = np.random.rand(nDots) * 360
        dotsRadius = (np.random.rand(nDots) ** 0.5) * 2
        speed = np.random.rand(nDots) * self.maxSpeed
        return [dotsTheta,dotsRadius,speed]
        
    def updateTrialAttributes(self,dir=None,coherence=None,color_proportion=None):
        if dir:
            assert dir in ['in','out']
            self.dir = dir
        if coherence:
            assert 0 <= coherence <= 1
            self.coherence = coherence
        if color_proportion:
            assert 0 <= color_proportion <= 1
            self.color_proportion = color_proportion
        
        
    def updateDotsPosition(self):
        if self.dir == 'out':
            speed = self.speed
        elif self.dir == 'in':
            speed = -1 * self.speed
        # update radius
        self.dotsRadius += speed
        # random radius where radius too large
        outFieldDots = np.logical_or(self.dotsRadius 
                                        >= self.radius,  self.dotsRadius<.1)
        self.dotsRadius[outFieldDots] = random.rand(sum(outFieldDots))*(self.radius-.1)+.1
        dotsX, dotsY = pol2cart(self.dotsTheta, self.dotsRadius)
        dotsX *=  float(self.win.size[1])/self.win.size[0]  # to account for wider aspect ratio
        self.dots.xys = np.array([dotsX, dotsY]).transpose()

    def updateDotsColor(self):
        color = self.baseColors[0]*self.color_proportion \
                + self.baseColors[1]*(1-self.color_proportion)
        colors = color.reshape(1,3).repeat(self.nElements,0)
        self.dots.setColors(colors)
        
    def draw(self):
        self.updateDotsColor()
        self.updateDotsPosition()
        self.dots.draw()
        self.win.flip()
        

def get_win(screen=0,fullscr=True):
    return  visual.Window([1024,768], color=[-1,-1,-1], allowGUI=False, fullscr=fullscr, 
                                     monitor='testMonitor', units='norm',  screen=screen) 
    
win = get_win(fullscr=True)
colors = np.array([[0,0,1],[0,1,0]])
stim = RadialOpticFlow(win,.03, colors, 1, sizes = .008, nElements = 4000)
color_proportion = 0
while True:
    keys=event.getKeys()
    if 'q' in keys:
        break
    if keys != []:
        keys=event.waitKeys()
            
    stim.updateTrialAttributes(color_proportion = np.sin(color_proportion)*.5+.5)
    color_proportion+=.02
    if color_proportion%6 > 3:
        stim.updateTrialAttributes(dir = 'in')
    else:
        stim.updateTrialAttributes(dir = 'out')
    stim.draw()
win.close()

"""