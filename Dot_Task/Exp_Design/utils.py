import pickle
from glob import glob
import numpy as np
import os
from psychopy import monitors

def get_trackers(subjid):
    file_dir = os.path.dirname(__file__)
    try:
        motion_file = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                        subjid, '*%s*motion*' % subjid)))[-1]
        motion_data = pickle.load(open(motion_file,'rb'))
        motion_trackers = motion_data['trackers']
        print('Found Motion Trackers. Loading from file: %s\n' % motion_file)
    except IndexError:
        motion_trackers = {}
    try:
        orientation_file = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                       subjid, '*%s*orientation*' % subjid)))[-1]
        orientation_data = pickle.load(open(orientation_file,'rb'))
        orientation_trackers = orientation_data['trackers']
        print('Found Orientation Trackers. Loading from file: %s\n' % orientation_file)
    except IndexError:
        orientation_trackers = {}
    return {'motion': motion_trackers, 
            'orientation': orientation_trackers}

def get_difficulties(subjid=None, trackers=None):
    assert trackers or subjid
    if trackers is None:
        trackers = get_trackers(subjid)
    difficulties = {}
    for dim, value in trackers.items():
        difficulties[dim] = {}
        for subkey, subvalue in value.items():
            difficulties[dim][subkey] = subvalue.mean()
    return difficulties
    
def get_monitor(distance=30, width=30):  
    monitor = monitors.Monitor('test')
    monitor.setDistance(60)
    monitor.setSizePix([2560,1440])
    monitor.setWidth(60)
    return monitor
