# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:22:54 2014

@author: admin
"""

import cPickle
import numpy as np
import pandas as pd
from scipy.stats import norm

def load_data(datafile, mode = 'train'):
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
    stim_df = pd.DataFrame(df.stim.tolist())
    df.drop('stim', axis=1, inplace=True)
    df = pd.concat([df,stim_df], axis=1)
    
    
    return (taskinfo, df)

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