from psychopy import visual, core, event
from psychopy.visual.dot import DotStim
from psychopy.tools.attributetools import setAttribute
import pyglet
pyglet.options['debug_gl'] = False
import ctypes
GL = pyglet.gl
import numpy as np
from skimage.color import lab2rgb

def pixel_lab2rgb(lst):
    lst = [float(x) for x in lst]
    return lab2rgb([[(lst)]]).flatten()
    
    
class ColorDotStim(DotStim):
    """
    Motion/Color Dot Stim
    """
    def __init__(self, win, color_proportion, **kwargs):
        self.color_proportion = color_proportion
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
            n_color1 = int(self.nDots*self.color_proportion)
            n_color2 = self.nDots - n_color1
            colors = np.array([color1 for _ in range(n_color1)] + [color2 for _ in range(n_color2)]).astype(ctypes.c_float)
            colors_gl = colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            GL.glColorPointer(4, GL.GL_FLOAT,0, colors_gl)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            
            #back to default code
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.nDots)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
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

    def setColorProportion(self, val, op='', log=None):
        """Usually you can use 'stim.attribute = value' syntax instead,
        but use this method if you need to suppress the log message
        """
        setAttribute(self, 'color_proportion', val, log, op)

        
class ColorDensityStim(DotStim):
    """
    Color dot stim with an outer ring that can be more/less dense or have higher/
    lower contrast
    """
    def __init__(self, win, color_proportion, outer_proportion, outer_opacity, **kwargs):
        self.color_proportion = color_proportion
        self.outer_proportion = outer_proportion
        self.outer_opacity = outer_opacity
        super(ColorDensityStim, self).__init__(win = win, **kwargs)
        
    def _newDotsXY(self, nDots):
        """Returns a uniform spread of dots, according to the
        fieldShape and fieldSize

        usage::

            dots = self._newDots(nDots)

        """
        outer_nDots = int(nDots*self.outer_proportion)
        # make more dots than we need and only use those within the circle
        if self.fieldShape == 'circle':
            inner_dots = []
            while len(inner_dots)==0:
                # repeat until we have enough; fetch twice as many as needed
                new = np.random.uniform(-1, 1, [nDots * 2, 2])
                inCircle = (np.hypot(new[:, 0], new[:, 1]) < 1)
                if sum(inCircle) >= nDots:
                    inner_dots = new[inCircle, :][:nDots, :] * 0.5
            outer_dots = []
            while len(outer_dots)==0:
                # repeat until we have enough; fetch twice as many as needed
                new = np.random.uniform(-2**.5, 2**.5, [outer_nDots * 4, 2])
                inCircle = np.logical_and(np.hypot(new[:, 0], new[:, 1]) > 1,
                            np.hypot(new[:, 0], new[:, 1]) < 2**.5)
                if sum(inCircle) >= outer_nDots:
                    outer_dots = new[inCircle, :][:outer_nDots, :] * 0.5
            dots = np.vstack([inner_dots,outer_dots])
            return dots
        else:
            return np.random.uniform(-0.5, 0.5, [nDots, 2])
            
    def _update_dotsXY(self):
        """The user shouldn't call this - its gets done within draw().
        """

        # Find dead dots, update positions, get new positions for
        # dead and out-of-bounds
        # renew dead dots
        outer_nDots = int(self.nDots*self.outer_proportion)
        # make more dots than we need and only use those within the circle
        if self.fieldShape == 'circle':
            inner_dots = []
            while len(inner_dots)==0:
                # repeat until we have enough; fetch twice as many as needed
                new = np.random.uniform(-1, 1, [self.nDots * 2, 2])
                inCircle = (np.hypot(new[:, 0], new[:, 1]) < 1)
                if sum(inCircle) >= self.nDots:
                    inner_dots = new[inCircle, :][:self.nDots, :] * 0.5
            outer_dots = []
            while len(outer_dots)==0:
                # repeat until we have enough; fetch twice as many as needed
                new = np.random.uniform(-2**.5, 2**.5, [outer_nDots * 4, 2])
                inCircle = np.logical_and(np.hypot(new[:, 0], new[:, 1]) > 1,
                            np.hypot(new[:, 0], new[:, 1]) < 2**.5)
                if sum(inCircle) >= outer_nDots:
                    outer_dots = new[inCircle, :][:outer_nDots, :] * 0.5
            dots = np.vstack([inner_dots,outer_dots])
            
        self._verticesBase = dots

        # update the pixel XY coordinates in pixels (using _BaseVisual class)
        self._updateVertices()
        
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
            color1 = self.color[0]
            color2 = self.color[1]
            total_dots = int(self.nDots*(1+self.outer_proportion))
            n_color1 = int(total_dots*self.color_proportion)
            n_color2 = total_dots - n_color1
            colors = np.array([color1 for _ in range(n_color1)] 
                               + [color2 for _ in range(n_color2)])
            np.random.shuffle(colors)
            # add opacities:
            opacity = [self.opacity]*self.nDots+[self.outer_opacity]*(total_dots-self.nDots)
            colors = np.array([np.append(c,opacity[i]) for i,c in enumerate(colors)]).astype(ctypes.c_float)
            colors_gl = colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            GL.glColorPointer(4, GL.GL_FLOAT,0, colors_gl)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            
            #back to default code
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDrawArrays(GL.GL_POINTS, 0, total_dots)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        else:
            # we don't want to do the screen scaling twice so for each dot
            # subtract the screen centre
            initialDepth = self.element.depth
            for pointN in range(0, total_dots):
                _p = self.verticesPix[pointN, :] + self.fieldPos
                self.element.setPos(_p)
                self.element.draw()
            # reset depth before going to next frame
            self.element.setDepth(initialDepth)
        GL.glPopMatrix()

    def setColorProportion(self, val, op='', log=None):
        """Usually you can use 'stim.attribute = value' syntax instead,
        but use this method if you need to suppress the log message
        """
        setAttribute(self, 'color_proportion', val, log, op)   
        
        
        
        
class TwoColorStim(DotStim):
    """
    Color dot stim with an outer ring that can be more/less dense or have higher/
    lower contrast. Expects colors in lab format
    """
    def __init__(self, win, color_proportions, colors, **kwargs):
        assert len(np.array(colors).flatten()) == 12, \
            "Must specify 4 lab colors"
        self.colors = np.array(colors)
        self.color_proportions = np.array(color_proportions)
        assert not np.any(abs(self.color_proportions-1)>1), \
            "Color proportions must be between 0 and 1"   
        super(TwoColorStim, self).__init__(win = win, **kwargs)
        

            
    def _update_dotsXY(self):
        """The user shouldn't call this - its gets done within draw().
        """
        dots = self._newDotsXY(self.nDots)
        self._verticesBase = dots
        # update the pixel XY coordinates in pixels (using _BaseVisual class)
        self._updateVertices()
        
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
            colors = []
            for i,dimension in enumerate(self.colors):
                dimension_color = dimension[0]*(self.color_proportions[i]) \
                                + dimension[1]*(1-self.color_proportions[i])
                dimension_color = [list(pixel_lab2rgb(dimension_color))]
                colors += dimension_color*(self.nDots/2)
            colors = np.array(colors).astype(ctypes.c_float)
            
            colors_gl = colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            GL.glColorPointer(3, GL.GL_FLOAT,0, colors_gl)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            
            #back to default code
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.nDots)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
            
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

    def setColorProportion(self, val, op='', log=None):
        """Usually you can use 'stim.attribute = value' syntax instead,
        but use this method if you need to suppress the log message
        """
        val = np.array(val)
        assert not np.any(abs(val-1)>1), \
            "Color proportions must be between 0 and 1"
        setAttribute(self, 'color_proportions', val, log, op)   
        

# access methods

def getColorDotStim(win, motion_coherence = .5, color_proportion = .5, direction = 0, colors = None):
    if colors == None:
        colors = [(1.0,0.0,0.0), (0.0,0.8,0.8)]
    dots = ColorDotStim(win, color_proportion, nDots = 500, dotSize = 4, signalDots = 'different', fieldShape = 'circle',
                          fieldSize = 15, speed = .05,  coherence = motion_coherence,  dir = direction,
                          color = colors, opacity = 1)
    return dots

def getColorDensityStim(win, color_proportion=.5, outer_proportion=1.8, 
                   opacity=1, outer_opacity=1, colors = None):
    if colors == None:
        colors = [(1.0,0.0,0.0), (0.0,0.8,0.8)]
    dots = ColorDensityStim(win, color_proportion, nDots = 1000, dotSize = 4, 
                            signalDots = 'different', fieldShape = 'circle',
                            fieldSize = 15, color = colors, opacity = opacity,
                            outer_proportion = outer_proportion, 
                            outer_opacity = outer_opacity)
    return dots
    

def getTwoColorStim(win, color_proportions=[.5,.5], 
                   opacity=1, colors = None):
    if colors == None:
        colors = [[(80,100,80), (80,-100,80)], [(80,0,100), (80,0,-100)]]
    dots = TwoColorStim(win, colors=colors, color_proportions=color_proportions,
                        nDots=1000, dotSize=4, signalDots='different', 
                        fieldShape='circle',fieldSize=15)
    return dots
        
