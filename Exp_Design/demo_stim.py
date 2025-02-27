from flowstim import OpticFlow
from hsluv import hsluv_to_rgb
import numpy as np
from psychopy import visual, event, core, monitors

from utils import pixel_lab2rgb, get_monitor

monitor = get_monitor()
def get_win(screen=0,fullscr=True):       
    return  visual.Window([1200,1200], color=[-1,-1,-1], allowGUI=False, fullscr=fullscr, 
                                     monitor=monitor, units='deg',  screen=screen,
                                     allowStencil=True) 
def define_aperture(win):
     # define aperture
    aperture_size = 30
    aperture_vertices = visual.Aperture(win, size=aperture_size).vertices
    #ratio = float(win.size[1])/win.size[0]
    #aperture_vertices[:,0]*=ratio
    aperture = visual.Aperture(win, size=aperture_size, shape = aperture_vertices)
    return aperture

def presentTextToWindow(win, text,size=.15):
    """ present a text message to the screen
    return:  time of completion
    """

    textStim=visual.TextStim(win, text=text,font='BiauKai',
                            height=size,color=[1]*3, colorSpace=u'rgb',
                            opacity=1,depth=0.0,
                            alignHoriz='center',wrapWidth=50)
    textStim.draw()
    win.flip()
    event.waitKeys()
        
# window, apeture, and stim setup           
win = get_win(fullscr=True)
aperture = define_aperture(win)
aperture.enable()
# potential color spaces
base_color = [1,1,1] # white
colors = np.array([[75,0,128],[75,0,-128]])
#colors = np.array([[75,128,75],[75,-128,75]])
# stim parameters
base_speed =.01
base_ori = 45
height = 1.5
ratio = .3
stim = OpticFlow(win,base_speed, color = base_color, mask='bar',
                 sizes=[height*ratio, height], 
                 nElements = 600, oris=base_ori)


 
# *************************************************************************
# DEMO Different Conditions In/Out + Color Space
# *************************************************************************

# defines the demo conditions to move through
presentation_order = [('in',.15, 45),
                      ('out',.85, 45),
                      ('out',.15, 315),
                      ('in',.85, 315)]   
index=0
presentTextToWindow(win,'Demo Start')
while True:
    keys=event.getKeys()
    if 'q' in keys:
        break
    elif 'space' in keys:
        index = (index+1)%len(presentation_order)
    elif 'r' in keys:
        index=0
    elif keys != []:
        keys=event.waitKeys()
    
    # update trial attributes
    dir = presentation_order[index][0]
    ori = presentation_order[index][2]
    color_proportion = presentation_order[index][1]
    color = colors[0]*color_proportion \
                + colors[1]*(1-color_proportion)
    # convert LAB color to RGB
    color = pixel_lab2rgb(color)
    stim.updateTrialAttributes(dir=dir, color=color, ori=ori)
    stim.draw()
    
# *************************************************************************
# DEMO Orientation changes
# *************************************************************************

presentTextToWindow(win,'Orientation Demo')
# duration per episode
duration = 3
presentation_order = [('in', 90),
                      ('in', 0),
                      ('out', 90),
                      ('out', 0)]
# show changes of color and speed
for dir, oe in presentation_order:
    stim.updateTrialAttributes(dir=dir, color=base_color)
    stim_clock = core.Clock()
    while True:
        percent_complete = (stim_clock.getTime()%duration)/duration
        # change speed
        ori = base_ori*(1-percent_complete) + oe*percent_complete
        stim.updateTrialAttributes(ori=ori)
        # at the end of a cycle wait for key press
        if .98 < percent_complete:
            win.flip()
            keys=event.waitKeys()
            stim_clock.reset()
            if 'q' in keys: 
                break
        # wait for keys
        keys=event.getKeys()
        if 'q' in keys:
            break
        elif keys != []:
            keys=event.waitKeys()
        stim.draw()


# *************************************************************************
# DEMO Speed changes
# *************************************************************************

presentTextToWindow(win,'Speed Demo')
# duration per episode
duration = 3
presentation_order = [('in',.25),
                      ('in',.02),
                      ('out',.25),
                      ('out',.02)]
# show changes of color and speed
for dir, se in presentation_order:

    stim.updateTrialAttributes(dir=dir, ori=base_ori, color=base_color)
    
    stim_clock = core.Clock()
    while True:
        percent_complete = (stim_clock.getTime()%duration)/duration
        # change speed
        speed = base_speed*(1-percent_complete) + se*percent_complete
        stim.updateTrialAttributes(speed=speed)
        # at the end of a cycle wait for key press
        if .98 < percent_complete:
            win.flip()
            keys=event.waitKeys()
            stim_clock.reset()
            if 'q' in keys: 
                break
        # wait for keys
        keys=event.getKeys()
        if 'q' in keys:
            break
        elif keys != []:
            keys=event.waitKeys()
        stim.draw()
        
"""
# *************************************************************************
# DEMO Color Spectrum
# *************************************************************************
presentTextToWindow(win,'Color Spectrum')
spectrum_stim = visual.Rect(win,units = 'norm', width = .6, height= .5)
color_direction=1
color_proportion = .02
while True:
    keys=event.getKeys()
    if 'q' in keys:
        break
    color_proportion+=(.005*color_direction)
    if color_proportion > .98 or color_proportion < .02:
        color_direction*=-1    
    color = colors[0]*color_proportion \
                + colors[1]*(1-color_proportion)
    color = pixel_lab2rgb(color)
    spectrum_stim.setFillColor(color)
    spectrum_stim.setLineColor(color)
    spectrum_stim.draw()
    win.flip()
    
# *************************************************************************
# DEMO Color changes
# *************************************************************************
presentTextToWindow(win,'Color Demo')
# duration per episode
duration = 3
stim.updateTrialAttributes(speed=base_speed)
presentation_order = [('in',.15, .3),
                      ('in',.15, 0),
                      ('out',.85, 1),
                      ('out',.85, .7)]
# show changes of color and speed
for dir,start_proportion, end_proportion in presentation_order:
    # update trial attributes
    cs = colors[0]*start_proportion \
                + colors[1]*(1-start_proportion)
    ce = colors[0]*end_proportion \
                + colors[1]*(1-end_proportion)
    # convert LAB color to RGB
    color = pixel_lab2rgb(cs)
    stim.updateTrialAttributes(dir=dir, color=color)
    
    stim_clock = core.Clock()
    while True:
        percent_complete = (stim_clock.getTime()%duration)/duration
        # smoothly move color over the duration
        color = cs*(1-percent_complete) + ce*percent_complete
        color = pixel_lab2rgb(color)
        stim.updateTrialAttributes(color=color)
        # at the end of a cycle wait for key press
        if .98 < percent_complete:
            win.flip()
            keys=event.waitKeys()
            stim_clock.reset()
            if 'q' in keys: 
                break
        # wait for keys
        keys=event.getKeys()
        if 'q' in keys:
            break
        elif keys != []:
            keys=event.waitKeys()
        stim.draw()
"""        
win.close()      
