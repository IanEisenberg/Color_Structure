import cPickle
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from psychopy import monitors

# convert from LAB to RGB space
from skimage.color import lab2rgb
import os

def get_difficulties(subject_code):
    file_dir = os.path.dirname(__file__)
    try:
        motion_file = glob(os.path.join(file_dir,'..','Data','RawData',
                                        '*%s*motion*' % subject_code))[-1]
        motion_data = cPickle.load(open(motion_file,'r'))
        motion_difficulties = {k:v.mean() for k,v in motion_data['trackers'].items()}
    except IndexError:
        motion_difficulties = {}
    try:
        color_file = glob(os.path.join(file_dir,'..','Data','RawData',
                                       '*%s*color*' % subject_code))[-1]
        color_data = cPickle.load(open(color_file,'r'))
        color_difficulties = {k:v.mean() for k,v in color_data['trackers'].items()}
    except IndexError:
        color_difficulties = {}
    return motion_difficulties, color_difficulties

def get_trackers(subject_code):
    file_dir = os.path.dirname(__file__)
    try:
        motion_file = glob(os.path.join(file_dir,'..','Data','RawData',
                                        '*%s*motion*' % subject_code))[-1]
        motion_data = cPickle.load(open(motion_file,'r'))
        motion_trackers = motion_data['trackers']
    except IndexError:
        motion_trackers = {}
    try:
        color_file = glob(os.path.join(file_dir,'..','Data','RawData',
                                       '*%s*color*' % subject_code))[-1]
        color_data = cPickle.load(open(color_file,'r'))
        color_trackers = color_data['trackers']
    except IndexError:
        color_trackers = {}
    return motion_trackers, color_trackers

def get_monitor(distance=30, width=30):  
    monitor = monitors.Monitor('test')
    monitor.setDistance(60)
    monitor.setSizePix([2560,1440])
    monitor.setWidth(60)
    return monitor

def pixel_lab2rgb(lst):
    lst = [float(x) for x in lst]
    return lab2rgb([[(lst)]]).flatten()*2-1
            
def plot_weibull(alpha, beta=3.5, chance = .5):
    x = np.linspace(0,alpha*3,100)
    y = chance + (1.0-chance)*(1-np.exp( -(x/alpha)**(beta) ))
    plt.plot(x,y)
    return alpha
    

