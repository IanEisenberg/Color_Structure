from dot_stim import getColorDotStim,  getColorDensityStim,  getTwoColorStim
import numpy as np
from psychopy import visual, core, event

def get_win():
    return  visual.Window([1200,800], color = [-.8,-.8,-.8], allowGUI=False, fullscr=False, 
                                     monitor='testMonitor', units='deg') 
def display_stim(win, stim, n):
    for _ in range(n):
        stim.draw()
        win.flip()
        keys = event.getKeys(keyList=['s', 'l'])
        if keys != []:
            break
        core.wait(.05)
 

# TwoColor Static Stim  
win = get_win()
stim = getTwoColorStim(win, color_proportions = [.4,0])
display_stim(win,stim,50)
win.close()

# TwoColor Dynamic Static Stim  
def get_color_steps(color_ranges, n_steps):
    color_steps = []
    for color_range in color_ranges:
        color_steps.append(np.linspace(*color_range, num=n_steps))
    color_steps = np.array(color_steps)
    return color_steps
    
n_steps = 50
n_frames = 100
color_steps = get_color_steps([[0,1],[0,0]],n_steps)
stim = getTwoColorStim(win, color_proportions = color_steps[:,0])
win = get_win()
# start display
display_stim(win,stim,10)
# change color
for step in range(1,n_steps):
    display_stim(win,stim,n_frames/n_steps)
    stim.setColorProportion(color_steps[:,step])
# end display
display_stim(win,stim,10)

win.close()

      