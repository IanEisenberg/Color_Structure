#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 00:29:13 2017

@author: ian
"""
from psychopy import visual, core, event
import sys,os
import json
import yaml
import datetime
import subprocess


win = visual.Window([800,600], color = [-.5,-.5,-.5], allowGUI=False, fullscr=False, 
                                 monitor='testMonitor', units='deg')



    
def presentDotStim(win):
    dots = visual.DotStim(win, nDots = 1200, dotSize = 3, signalDots = 'different', fieldShape = 'circle',
                          fieldSize = 15, speed = .1, color = (1,1,1), coherence = .5,  dir = 0)
    return dots
    
dots = presentDotStim(win)

for _ in range(5):
    dots.draw()
    win.flip()
win.close()



import pyglet
pyglet.options['debug_gl'] = False
import ctypes
GL = pyglet.gl
import numpy as np

#reference: http://stackoverflow.com/questions/15603931/using-numpy-arrays-and-ctypes-for-opengl-calls-in-pyglet
self= dots
colors = np.array([(0.0,0.0,1.0)]*(self.nDots/2) + [(1.0,0.0,0.0)]*(self.nDots/2)).astype(ctypes.c_float)
colors_gl = colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

GL.glPushMatrix()  # push before drawing, pop after
win.setScale('pix')
GL.glPointSize(self.dotSize)

# load Null textures into multitexteureARB - they modulate with
# glColor
GL.glActiveTexture(GL.GL_TEXTURE0)
GL.glEnable(GL.GL_TEXTURE_2D)
GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
GL.glActiveTexture(GL.GL_TEXTURE1)
GL.glEnable(GL.GL_TEXTURE_2D)
GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

CPCD = ctypes.POINTER(ctypes.c_double)
GL.glVertexPointer(2, GL.GL_DOUBLE, 0,
                   self.verticesPix.ctypes.data_as(CPCD))
GL.glColorPointer(3, GL.GL_FLOAT,0, colors_gl)
desiredRGB = self._getDesiredRGB(self.rgb, self.colorSpace,
                                 self.contrast)

GL.glEnableClientState(GL.GL_COLOR_ARRAY)
GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
GL.glDrawArrays(GL.GL_POINTS, 0, self.nDots)
GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
GL.glPopMatrix()

win.flip()