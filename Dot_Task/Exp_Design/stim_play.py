from dot_stim import getColorDotStim,  getColorDensityStim,  getTwoColorStim
import numpy as np
from psychopy import visual, core, event

# helper functions
def get_win(screen=0,fullscr=False):
    return  visual.Window([1024,378], color=[-.8,-.8,-.8], allowGUI=False, fullscr=fullscr, 
                                     monitor='testMonitor', units='deg', screen=screen) 
def display_stim(win, stim, n, frame_delay=.05):
    for _ in range(n):
        stim.draw()
        win.flip()
        keys=event.getKeys()     
        if 'q' in keys:
            break
        if keys != []:
            print keys
            keys=event.waitKeys()
            if 'q' in keys:
                break
        if frame_delay>0:
            core.wait(frame_delay)
    return keys
 
def get_color_steps(color_ranges, n_steps):
    color_steps=[]
    for color_range in color_ranges:
        color_steps.append(np.linspace(*color_range, num=n_steps))
    color_steps=np.array(color_steps)
    return color_steps

# ****************************************************************************
params={'nDots': 2000, 'fieldSize': 15}

# TwoColor Stim Dimensions
# Show base stim in sequence
colors=[[(80,0,-80), (80,80,0)], [(80,-80,0), (80,0,80)]]
base_stims=[]
win=get_win(0, False)
for proportion in [[0,0],[1,0],[1,1],[1,0]]:
    stim=getTwoColorStim(win, colors=colors, 
                           color_proportions=proportion, **params)
    base_stims.append(stim)
    
for stim in base_stims:
    display_stim(win,stim,25)
win.close()

# Show each dimensions changing in sequence
# set up display variables
n_steps=50
n_frames=100
color_steps=get_color_steps([[0,1],[0,1]],n_steps)
# make both populations of dots the same color
colors=[[[(80,0,-80), (80,80,0)], [(80,0,-80), (80,80,0)]],
          [[(80,-80,0), (80,0,80)], [(80,-80,0), (80,0,80)]]]
win=get_win(0)
for color in colors:
    stim=getTwoColorStim(win, color_proportions=color_steps[:,0], 
                         colors=color, **params)
    # change color 
    for step in range(1,n_steps):
        keys=display_stim(win,stim,n_frames/n_steps)
        if 'q' in keys:
            break
        stim.setColorProportion(color_steps[:,step])
win.close()
    

# demonstrate harder stim
colors=[[(80,0,-80), (80,80,0)], [(80,-80,0), (80,0,80)]]
stim1=getTwoColorStim(win, colors=colors, color_proportions=[.45,0], **params)
stim2=getTwoColorStim(win, colors=colors, color_proportions=[.55,0], **params)

win=get_win()
display_stim(win,stim1,50)
display_stim(win,stim2,50)
win.close()


# ****************************************************************************
# Different kind of stim. Just circles

from skimage.color import lab2rgb

def pixel_lab2rgb(lst):
    lst=[float(x) for x in lst]
    return lab2rgb([[(lst)]]).flatten()

class circleStim(object):
    
    def __init__(self,win,radius,colors,color_proportion):
        inner_color_lab = np.array(colors[0][0])*color_proportion[0] \
                        + np.array(colors[0][1])*(1-color_proportion[0])
        outer_color_lab = np.array(colors[1][0])*color_proportion[1] \
                        + np.array(colors[1][1])*(1-color_proportion[1])
    
        inner_color = pixel_lab2rgb(inner_color_lab)
        outer_color = pixel_lab2rgb(outer_color_lab)
        self.inner_circle = visual.Circle(win, radius=radius, 
                                          fillColor=inner_color, lineColor=inner_color)
        self.outer_circle = visual.Circle(win, radius=radius*2**.5, 
                                          fillColor=outer_color, lineColor=outer_color)
        
    def draw(self):
        self.outer_circle.draw()
        self.inner_circle.draw()
        


    
colors=[[(80,0,-80), (80,80,0)], [(80,-80,0), (80,0,80)]]
radius = 2
win=get_win(0)
stim = circleStim(win,radius,colors,[.55,0])
stim.draw()
win.flip()
win.close()   
