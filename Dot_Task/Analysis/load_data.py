# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:22:54 2014

@author: admin
"""

import cPickle
from glob import glob
import numpy as np
import os
import pandas as pd
from scipy.stats import norm

def load_data(datafile):
    """
    Load a temporal structure task data file. Cleans up the raw data (returns
    the first action/rt, removes trials without a response). Returns the global
    taskinfo, the cleaned up data and a new dataset for analysis (with some
    variables removed, and some created)
    
    Finally saves the data as csv files
    """
    f=open(datafile, 'r')
    loaded_pickle = cPickle.load(f)
    data = loaded_pickle['taskdata']
    taskinfo = loaded_pickle['taskinfo']
    
    
    #Load data into a dataframe
    df = pd.DataFrame(data)
    # separate stim attributes
    stim_df = pd.DataFrame(df.query('rt==rt').stim.tolist())
    stim_df.index = df.query('rt==rt').index
    df.drop('stim', axis=1, inplace=True)
    df = pd.concat([df,stim_df], axis=1)
    return (taskinfo, df)


def preproc_threshold_data(df):
    df.insert(0, 'binarized_response', df.response.replace({'up':1, 'down':0, 
                                                            'right': 1, 'left': 0}))
              
def load_threshold_data(subj_code, dim='motion'):
    file_dir = os.path.dirname(__file__)
    assert dim in ['motion','orientation']
    files = sorted(glob(os.path.join(file_dir,'..','Data','RawData',
                                     '%s_*%s*' % (subj_code, dim))))
    if len(files) > 0:
        datafile = pd.DataFrame()
        for i, filey in enumerate(files):
            taskinfo,df = load_data(filey)
            df.insert(0, 'Session', i)
            preproc_threshold_data(df)
            datafile = pd.concat([datafile, df])
        # reorganize
        datafile.reset_index(drop=True, inplace=True)
        return taskinfo, datafile
    else:
        print('No %s files found for subject %s!' % (dim, subj_code))
        return None, None

def preproc_data(traindata, testdata, taskinfo, dist = norm):
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