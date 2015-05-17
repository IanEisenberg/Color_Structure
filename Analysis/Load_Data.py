# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:22:54 2014

@author: admin
"""

import yaml
import numpy as np
import pandas as pd

def load_data(datafile, name, mode = 'train'):
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
    df.rt[:] = rts
    df.response[:] = responses
    #Remove missed trials:
    df = df[df.rt != 999]
    df = df.reset_index(drop=True)
    #change responses to numerical values if need be:
    if type(df['response'].loc[df.index[0]]) == str:
        df['response'] = [taskinfo['action_keys'].index(response) for response in df.response]
    

    #Create a separate analysis dataframe
    if mode == 'train':
        dfa = df.drop(['FBonset', 'FBDuration', 'actualFBOnsetTime', 'actualOnsetTime', 'onset', 
                        'reward', 'punishment','stimulusCleared'],1)
    elif mode == 'test':
        dfa = df.drop(['FBonset', 'FB', 'FBDuration', 'actualOnsetTime', 'onset', 
                       'reward', 'punishment','stimulusCleared'],1)
                  
    dfa['rep_resp'] = [dfa.response.shift(1)[i] == dfa.response[i] for i in dfa.index]
    dfa['switch'] = abs(dfa.ts.diff())
    dfa['con_1dim'] = [int(dfa.response[i] == dfa.stim[i][0]) for i in dfa.index]
    dfa['con_2dim'] = [int(dfa.response[i] == dfa.stim[i][1]) for i in dfa.index]
    dfa['subj_ts'] = [int(response in [2,3]) for response in dfa.response]
    dfa['subj_switch'] = abs(dfa.subj_ts.diff())
    dfa['correct'] = [dfa.response[i] == dfa.stim[i][dfa.ts[i]] for i in dfa.index]
    dfa['stim_conform'] = [dfa.response.loc[i] in dfa.stim.loc[i] for i in dfa.index]
    dfa = dfa.convert_objects(convert_numeric = True)
    
    return (taskinfo, dfa)

