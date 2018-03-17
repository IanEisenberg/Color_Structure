import cPickle
from glob import glob
import numpy as np
import os
from psychopy import monitors


def get_difficulties(subjid):
    file_dir = os.path.dirname(__file__)
    try:
        motion_file = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                        subjid, '*%s*motion*' % subjid)))[-1]
        motion_data = cPickle.load(open(motion_file,'r'))
        speed_difficulties = {k:v.mean() for k,v in motion_data['trackers'].items()}
        print('Found Motion Difficulties. Loading from file: %s\n' % motion_file)
    except IndexError:
        speed_difficulties = {}
    try:
        orientation_file = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                       subjid, '*%s*orientation*' % subjid)))[-1]
        orientation_data = cPickle.load(open(orientation_file,'r'))
        orientation_difficulties = {k:v.mean() for k,v in orientation_data['trackers'].items()}
        print('Found Orientation Difficulties. Loading from file: %s\n' % orientation_file)
    except IndexError:
        orientation_difficulties = {}
    return {'motion': speed_difficulties, 
            'orientation': orientation_difficulties}

def get_trackers(subjid):
    file_dir = os.path.dirname(__file__)
    try:
        motion_file = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                        subjid, '*%s*motion*' % subjid)))[-1]
        motion_data = cPickle.load(open(motion_file,'r'))
        motion_trackers = motion_data['trackers']
        print('Found Motion Trackers. Loading from file: %s\n' % motion_file)
    except IndexError:
        motion_trackers = {}
    try:
        orientation_file = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                       subjid, '*%s*orientation*' % subjid)))[-1]
        orientation_data = cPickle.load(open(orientation_file,'r'))
        orientation_trackers = orientation_data['trackers']
        print('Found Orientation Trackers. Loading from file: %s\n' % orientation_file)
    except IndexError:
        orientation_trackers = {}
    return {'motion': motion_trackers, 
            'orientation': orientation_trackers}

def get_monitor(distance=30, width=30):  
    monitor = monitors.Monitor('test')
    monitor.setDistance(60)
    monitor.setSizePix([2560,1440])
    monitor.setWidth(60)
    return monitor


    

