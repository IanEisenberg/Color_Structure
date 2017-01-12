#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 00:29:13 2017

@author: ian
"""
from psychopy import visual, core
from psychopy.visual.dot import DotStim
from psychopy.tools.attributetools import setAttribute
import pyglet
pyglet.options['debug_gl'] = False
import ctypes
GL = pyglet.gl
import numpy as np

class ColorDotStim(DotStim):
    def __init__(self, win, color_coherence, **kwargs):
        self.color_coherence = color_coherence
        super(ColorDotStim, self).__init__(win = win, **kwargs)
        
    def draw(self, win=None):
        """Draw the stimulus in its relevant window. You must call
        this method after every MyWin.flip() if you want the
        stimulus to appear on that frame and then update the screen again.
        """
        if win is None:
            win = self.win
        self._selectWindow(win)

        self._update_dotsXY()

        GL.glPushMatrix()  # push before drawing, pop after

        # draw the dots
        if self.element is None:
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
            # set colors
            color1 = np.append(self.color[0],self.opacity)
            color2 = np.append(self.color[1],self.opacity) 
            n_color1 = int(self.nDots*self.color_coherence)
            n_color2 = self.nDots - n_color1
            colors = np.array([color1 for _ in range(n_color1)] + [color2 for _ in range(n_color2)]).astype(ctypes.c_float)
            colors_gl = colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            GL.glColorPointer(4, GL.GL_FLOAT,0, colors_gl)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            
            #back to default code
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.nDots)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        else:
            # we don't want to do the screen scaling twice so for each dot
            # subtract the screen centre
            initialDepth = self.element.depth
            for pointN in range(0, self.nDots):
                _p = self.verticesPix[pointN, :] + self.fieldPos
                self.element.setPos(_p)
                self.element.draw()
            # reset depth before going to next frame
            self.element.setDepth(initialDepth)
        GL.glPopMatrix()

    def setColorCoherence(self, val, op='', log=None):
        """Usually you can use 'stim.attribute = value' syntax instead,
        but use this method if you need to suppress the log message
        """
        setAttribute(self, 'color_coherence', val, log, op)
    
def getDotStim(win, motion_coherence = .5, color_coherence = .5, direction = 0, colors = None):
    if colors == None:
        colors = [(1.0,0.0,0.0), (0.0,0.8,0.8)]
    dots = ColorDotStim(win, color_coherence, nDots = 1000, dotSize = 4, signalDots = 'same', fieldShape = 'circle',
                          fieldSize = 15, speed = .05,  coherence = motion_coherence,  dir = direction,
                          color = colors, opacity = 1)
    return dots

def display_stim(win, stim, n):
    for _ in range(n):
        stim.draw()
        win.flip()
        core.wait(.015)
        
def play():
    win = visual.Window([1200,800], color = [-.8,-.8,-.8], allowGUI=False, fullscr=False, 
                                     monitor='testMonitor', units='deg')
    directions = [i%360 for i in range(720)]
    coherence = [(0.0+i)/(len(directions)-1) for i in range(len(directions))]
    dots = getDotStim(win, motion_coherence = .5, color_coherence = 0)
    
    display_stim(win,dots,50)
    for d,c in zip(directions,coherence):
        dots.draw()
        win.flip()
        core.wait(.015)
        dots.setDir(d)
        dots.setColorCoherence(c)
    display_stim(win,dots,50)
    
    win.close()
    return dots

dots = play()

    
     