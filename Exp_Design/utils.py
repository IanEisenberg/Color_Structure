import pickle
from glob import glob
import os
from psychopy import monitors
from Analysis.load_data import load_threshold_data
from Analysis.utils import fit_response_fun

def get_data_files(subjid):
    file_dir = os.path.dirname(__file__)
    motion_files = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                        subjid, '*%s*motion*' % subjid)))
    orientation_files = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                       subjid, '*%s*orientation*' % subjid)))
    return motion_files, orientation_files

def get_monitor(distance=30, width=30):  
    monitor = monitors.Monitor('test')
    monitor.setDistance(60)
    monitor.setSizePix([2560,1440])
    monitor.setWidth(60)
    return monitor

def get_response_curves(subjid):
    responseCurves = {}
    for dim in ['motion', 'orientation']:
        taskinfo, df = load_threshold_data(subjid, dim)
        assert df is not None, \
            print('No threshold data found for %s!' % subjid)
        init_estimate = .01 if dim=='motion' else 6
        responseCurve = fit_response_fun(responses=df.correct, 
                                         intensities=df.decision_var, 
                                         threshold_estimate=init_estimate,
                                         kind='lapseWeibull')
        responseCurves[dim] = responseCurve[0]
    return responseCurves

def get_trackers(subjid):
    motion_files, orientation_files = get_data_files(subjid)
    if len(motion_files) > 0:
        motion_file = motion_files[-1]
        motion_data = pickle.load(open(motion_file,'rb'))
        motion_trackers = motion_data['trackers']
        print('Found Motion Trackers. Loading from file: %s\n' % motion_file)
    else:
        motion_trackers = {}
    if len(orientation_files) > 0:
        orientation_file = orientation_files[-1]
        orientation_data = pickle.load(open(orientation_file,'rb'))
        orientation_trackers = orientation_data['trackers']
        print('Found Orientation Trackers. Loading from file: %s\n' % orientation_file)
    else:
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

def get_tracker_data(trackers, N=None):
    loaded_trackers = []
    responses = []
    intensities = []
    for k,v in trackers.items():
        if v not in loaded_trackers:
            responses += v.data
            intensities += v.intensities
        loaded_trackers.append(v)
    if N:
        responses = responses[-N:]
        intensities = intensities[-N:]
    return responses, intensities
        
def get_total_trials(trackers):
    loaded_trackers = []
    trials = 0
    for k,v in trackers.items():
        if v not in loaded_trackers:
            trials += len(v.data)
        loaded_trackers.append(v)
    return trials
    
def fix_trackers(subjid):
    motion_files, orientation_files = get_data_files(subjid)
    if len(motion_files) > 0:
        motion_file = motion_files[-1]
        motion_data = pickle.load(open(motion_file,'rb'))
        motion_trackers = motion_data['trackers']
        print('Found Motion Trackers. Loading from file: %s\n' % motion_file)
        taskdata = motion_data['taskdata']
        tracker = list(motion_trackers.values())[0]
        tracker.data = tracker.data[:128] + [i['FB'] for i in taskdata]       
        tracker.intensities = tracker.intensities[:128] + [i['decision_var'] for i in taskdata]
        pickle.dump(motion_data, open(motion_file, 'wb'))
    else:
        print('No motion file found')
    if len(orientation_files) > 0:
        ori_file = orientation_files[-1]
        ori_data = pickle.load(open(ori_file,'rb'))
        ori_trackers = ori_data['trackers']
        print('Found ori Trackers. Loading from file: %s\n' % ori_file)
        taskdata = ori_data['taskdata']
        tracker = list(ori_trackers.values())[0]
        tracker.data = tracker.data[:128] + [i['FB'] for i in taskdata]       
        tracker.intensities = tracker.intensities[:128] + [i['decision_var'] for i in taskdata]
        pickle.dump(ori_data, open(ori_file, 'wb'))
    else:
        print('No ori file found')