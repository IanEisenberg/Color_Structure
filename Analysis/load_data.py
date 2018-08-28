# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:22:54 2014

@author: admin
"""

import pickle
from glob import glob
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
    
def load_datafile(datafile):
    """
    Load a temporal structure task data file. Cleans up the raw data (returns
    the first action/rt, removes trials without a response). Returns the global
    taskinfo, the cleaned up data and a new dataset for analysis (with some
    variables removed, and some created)
    
    Finally saves the data as csv files
    """
    f=open(datafile, 'rb')
    loaded_pickle = pickle.load(f)
    data = loaded_pickle['taskdata']
    taskinfo = loaded_pickle['taskinfo']
    
    
    #Load data into a dataframe
    df = pd.DataFrame(data)
    # separate stim attributes
    stim_df = pd.DataFrame(df.stim.tolist())
    df = pd.concat([df,stim_df], axis=1)
    return (taskinfo, loaded_pickle['configfile'], df)
 
def load_datafiles(subjid, lookup_string, preproc_fun=None):
    file_dir = os.path.dirname(__file__)
    files = sorted(glob(os.path.join(file_dir,'..','Data','RawData',subjid,
                                     lookup_string)))
    session_i = 1
    run_i = 1
    last_date = None
    if len(files) > 0:
        data = pd.DataFrame()
        for i, filey in enumerate(files):
            date = os.path.basename(filey).split('_')[-2]
            if last_date and date != last_date:
                session_i += 1
                last_date = date
                run_i = 1
            taskinfo, configfile, df = load_datafile(filey)
            df.insert(0, 'run', run_i)
            df.insert(0, 'configfile', configfile)
            df.insert(0, 'session', session_i)
            df.insert(0, 'subjid', subjid)
            if preproc_fun:
                preproc_fun(df)
            data = pd.concat([data, df])
            run_i += 1
        # reorganize
        data.reset_index(drop=True, inplace=True)
        data.configfile = data.configfile.astype('category')
        return taskinfo, data, files
    else:
        print('No %s files found for subject %s!' % (lookup_string, subjid))
        return None, None, None

def load_cued_data(subjid, fmri=True):
    if fmri: 
        task = "*fmri_cued_dot_task*" 
    else: 
        task = "*cued_dot_task*"
    taskinfo, data, files = load_datafiles(subjid, task,
                                           preproc_fun=preproc_cued_data)
    return taskinfo, data, files

def load_threshold_data(subjid, dim="motion"):
    assert dim in ['motion','orientation']
    taskinfo, data, files = load_datafiles(subjid, f'*{dim}*', 
                                           preproc_fun=preproc_threshold_data)
    return taskinfo, data, files

def get_event_files(subjid, save=False):
    taskinfo, data, files = load_cued_data(subjid, fmri=True)
    run_starts = np.where(data.run!=data.run.shift(-1))[0]+1
    event_files = []
    start = 0
    median_rt = data.rt.median()
    event_file_locs = []
    for filey, end in zip(files, run_starts):
        run = data.iloc[start:end]
        event_run = preproc_fmri_data(run, median_rt)
        event_files.append(event_run)
        start=end
        if save:
            event_loc = filey.replace('RawData','EventsData')
            os.makedirs(os.path.dirname(event_loc), exist_ok=True)
            pickle.dump(event_run, open(event_loc, 'wb'))
            event_file_locs.append(event_loc)
    return event_files, sorted(event_file_locs)

def events_to_BIDS_dir(subjid, BIDS_dir):
    events, file_locs = get_event_files(subjid, True)
    BIDS_files = sorted(glob(os.path.join(BIDS_dir, '*%s*' % subjid,
                                          '*', 'func', '*bold*')))
    for events_file, fmri_file in zip(file_locs, BIDS_files):
        new_file = fmri_file.replace('bold.nii.gz', 'events.tsv')
        events_df = pickle.load(open(events_file,'rb'))
        events_df.to_csv(new_file, sep='\t')
        
# ****************************************************************************
# preproc functions
# ****************************************************************************
def preproc_cued_data(df):
    df.response.replace({'e': 'down', 'b': 'up', 'r': 'left', 'y': 'right'},
                        inplace=True)
    response_ts = np.where(df.response.isin(['up','down']), 
                           'motion', 'orientation')
    df.insert(df.columns.get_loc('ts'), 'response_ts',  response_ts)
    
def preproc_threshold_data(df):
    df.insert(df.columns.get_loc('response'), 'binarized_response', 
              df.response.replace({'up':1, 'down':0, 'right': 1, 'left': 0}))
    df.insert(df.columns.get_loc('speed_end'), 'speed_change', 
              df.speed_end-df.speed_start)
    df.insert(df.columns.get_loc('ori_end'), 'ori_change', 
              df.ori_end-df.ori_start)
    # drop missed RT
    assert np.mean(df.rt.isnull()) < .05, print('Many Missing Responses!')
    df.drop(df.query('rt!=rt').index, inplace=True)

def preproc_fmri_data(df, median_rt=None):
    if median_rt is None:
        median_rt = df.rt.median()
    event_file = []
    for i, row in df.iterrows():
        # determine if trial is junk
        junk = False
        if row.response_ts != row.ts or np.isnan(row.rt):
            junk = True
            
        generic = {'session': row.session,
                   'run': row.run,
                   'subjid': row.subjid,
                   'junk': junk}
        cue = {'onset': row.onset,
               'duration': row.cueDuration,
               'ts': row.ts,
               'type': 'cue'}
        stim = {'onset': row.onset+row.CSI,
                'duration': row.stimulusDuration,
                'motionDirection': row.motionDirection,
                'oriBase': row.oriBase,
                'oriDirection': row.oriDirection,
                'speedDirection': row.speedDirection,
                'ts': row.ts,
                'type': 'stimulus'}
        response_onset = row.onset+row.CSI+row.stimulusDuration+row.stimResponseInterval
        response = {'onset': response_onset,
                    'duration': 0, #impulse
                    'correct': row.correct,
                    'rt': row.rt - median_rt, 
                    'type': 'response'}
        for part in [cue, stim, response]:
            part.update(generic)
            event_file.append(part)
    return pd.DataFrame(event_file)
    
    
def preproc_context_data(traindata, testdata, taskinfo, dist = norm):
            """ Sets TS2 to always be associated with the 'top' of the screen (positive context values),
            creates a log_rt column and outputs task statistics during training
            :return: train_ts_dis, train_recursive_p, action_eps
            """
            #flip contexts if necessary
            states = taskinfo['states']
            ts_dists = {s['ts']: dist(**s['dist_args']) for s in states.values()}

            ts2_side = np.sign(ts_dists[1].mean())
            traindata['true_context'] = traindata['context']
            testdata['true_context'] = testdata['context']            
            traindata['context']*=ts2_side
            testdata['context']*=ts2_side
            #add log rt columns
            traindata['log_rt'] = np.log(traindata.rt)
            testdata['log_rt'] = np.log(testdata.rt)
            # What was the mean contextual value for each taskset during this train run?
            train_ts_means = list(traindata.groupby('ts').agg(np.mean).context)
            # Same for standard deviation
            train_ts_std = list(traindata.groupby('ts').agg(np.std).context)
            train_ts_dis = [norm(m, s) for m, s in zip(train_ts_means, train_ts_std)]
            train_recursive_p = 1 - traindata.switch.mean()
            action_eps = 1-np.mean([testdata['response'][i] in testdata['stim'][i] for i in testdata.index])
            return train_ts_dis, train_recursive_p, action_eps