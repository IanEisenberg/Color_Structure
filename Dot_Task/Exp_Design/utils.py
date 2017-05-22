import cPickle
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
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

def pixel_lab2rgb(lst):
    lst = [float(x) for x in lst]
    return lab2rgb([[(lst)]]).flatten()
            
from psychopy.data import _baseFunctionFit   

def fit_weibull(xx,yy, beta=3.5):
    _chance=.5
    class FitWeibull(_baseFunctionFit):
        """Fit a Weibull function (either 2AFC or YN)
        of the form::
    
            y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))
    
        and with inverse::
    
            x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)
    
        After fitting the function you can evaluate an array of x-values
        with ``fit.eval(x)``, retrieve the inverse of the function with
        ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
        (a list with ``[alpha, beta]``)
        """
        # static methods have no `self` and this is important for
        # optimise.curve_fit
        @staticmethod
        def _eval(xx, alpha):
            xx = np.asarray(xx)
            yy = _chance + (1.0 - _chance) * (1 - np.exp(-(xx / alpha)**beta))
            return yy
    
        @staticmethod
        def _inverse(yy, alpha):
            xx = alpha * (-np.log((1.0 - yy) / (1 - _chance))) ** (1.0 / beta)
            return xx
    fit = FitWeibull(xx,yy)
    return fit
    
def plot_weibull(alpha, beta=3.5, chance = .5):
    x = np.linspace(0,alpha*3,100)
    y = chance + (1.0-chance)*(1-np.exp( -(x/alpha)**(beta) ))
    plt.plot(x,y)
    return alpha
    

