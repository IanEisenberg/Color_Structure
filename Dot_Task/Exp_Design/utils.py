import pickle
from glob import glob
import numpy as np
import os
from psychopy import monitors
from Dot_Task.Analysis.load_data import load_threshold_data
from Dot_Task.Analysis.utils import fit_response_fun

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

def get_tracker_estimates(subjid=None, trackers=None):
    assert trackers or subjid
    if trackers is None:
        trackers = get_trackers(subjid)
    estimates = {}
    for dim, value in trackers.items():
        estimates[dim] = {}
        for subkey, subvalue in value.items():
            estimates[dim][subkey] = subvalue.mean()
    return estimates

def get_response_curve(subjid):
    responseCurves = {}
    for dim in ['motion', 'orientation']:
        taskinfo, df = load_threshold_data(subjid, dim)
        responseCurve = fit_response_fun(df, kind='lapseWeibull')
        responseCurves[dim] = responseCurve
    return responseCurves

def get_monitor(distance=30, width=30):  
    monitor = monitors.Monitor('test')
    monitor.setDistance(60)
    monitor.setSizePix([2560,1440])
    monitor.setWidth(60)
    return monitor
