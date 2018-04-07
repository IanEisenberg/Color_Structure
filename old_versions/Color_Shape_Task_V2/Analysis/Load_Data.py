# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:22:54 2014

@author: admin
"""

import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm, beta

def load_data(datafile, name):
    """
    Load a temporal structure task data file. Cleans up the raw data (returns
    the first action/rt, removes trials without a response). Returns the global
    taskinfo, the cleaned up data and a new dataset for analysis (with some
    variables removed, and some created)
    
    Finally saves the data as csv files
    """
    f=open(datafile)
    loaded_yaml = yaml.load(f)
    data = loaded_yaml['taskdata']
    taskinfo = loaded_yaml['taskinfo']
    
    
    #Load data into a dataframe
    df = pd.DataFrame(data)
    
    # Responses and RT's are stored as lists, though we only care about the first one.
    rts = np.array([x[0]  for x in df.rt.values])
    responses = [x[0] for x in df.response.values]
    df.loc[:,'rt'] = rts
    df.loc[:,'response'] = responses
    
    #Remove missed trials:
    df = df[df.rt != 999]
    df = df.reset_index(drop=True)

    

    #Create a separate analysis dataframe
    drop_cols = ['FBonset', 'FBDuration', 'actualFBOnsetTime', 
                 'actualOnsetTime', 'onset', 'displayFB',
                 'reward_amount', 'punishment_amount','stimulusCleared']
    dfa = df.drop(set(drop_cols) & set(df.columns),1)

    dfa['rep_resp'] = dfa.response.shift(1) == dfa.response
    dfa['switch'] = (dfa.ts.shift(1)==dfa.ts).astype(int)
    dfa['switch'].iloc[0]=False
    # label response as consistent with one task set or the other
    dfa['cons_TS1'] = [int(dfa.response[i] == dfa.stim[i][0]) for i in dfa.index]
    dfa['cons_TS2'] = [int(dfa.response[i] == dfa.stim[i][1]) for i in dfa.index]
    dfa['subj_ts'] = [int(response in [2,3]) for response in dfa.response]
    dfa['subj_switch'] = [int(dfa.subj_ts.shift(1)[i] != dfa.subj_ts[i]) for i in dfa.index]
    dfa['correct'] = [dfa.response[i] == dfa.stim[i][dfa.ts[i]] for i in dfa.index]
    dfa['stim_conform'] = [dfa.response.loc[i] in dfa.stim.loc[i] for i in dfa.index]
    dfa = dfa.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    
    return (taskinfo, df,dfa)

def preproc_data(traindata, testdata, taskinfo, dist = "norm"):
            """ Sets TS2 to always be associated with the 'top' of the screen (positive context values),
            creates a log_rt column and outputs task statistics during training
            :return: train_ts_dis, train_recursive_p, action_eps
            """
            if dist=="norm":
                dist=norm
            elif dist=="beta":
                dist=beta
            
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