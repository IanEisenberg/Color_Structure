# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:22:54 2014

@author: admin
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    #Reflects mapping of action keys to tasksets and stimuli
    stim_map = [(0,2),(1,3)]
    ts_map = [(0,1),(2,3)]
    
    #Load data into a dataframe
    df = pd.DataFrame(data)
    
    # Responses and RT's are stored as lists, though we only care about the first one.
    rts = np.array([x[0]  for x in df.rt.values])
    responses = [x[0] for x in df.response.values]
    df.rt[:] = rts
    df.response[:] = responses
    #Remove missed trials:
    df = df[df.response != 'NA']
    df = df.set_index(df.trial_count)
    #change response from strings to corresponding numbers.
    df.response = [taskinfo['action_keys'].index(key_press) for key_press in df.response]

    #Create a separate analysis dataframe
    if mode == 'train':
        dfa = df.drop(['FBonset', 'FBDuration', 'actualFBOnsetTime', 'actualOnsetTime', 'onset', 
                               'stimulusCleared'],1)
    elif mode == 'test':
        dfa = df.drop(['FBonset', 'FB', 'actualOnsetTime', 'onset', 
                               'stimulusCleared'],1)
    
    #Create new variables: taskset consistent, stim consistent

    # Taskset consistency means that the action was one of the two dictated by 
    # the current taskset, even if it isn't correct for the specific stimulus.
    # Stimulus consistency means that the action would be correct for the stimulus
    # in one of the two tasksets.
    dfa['con_ts'] = [any([stim[dfa.response[i]]==1 for stim in dfa.ts[i]]) for i in dfa.index]
    dfa['con_stim'] = [dfa.response[i] in stim_map[dfa.stim[i]] for i in dfa.index]
    # Presuming tasksets are learned, define a 'currently operating TS' variable
    #curr_ts is defined by the response, which corresponds to one of the two ts's defined
    #in ts_map
    dfa['curr_ts'] = [int(dfa.response[i] in ts_map[1]) for i in dfa.index]
    dfa['switch'] = [dfa.curr_ts.shift(1)[i] != dfa.curr_ts[i] for i in dfa.index]
    dfa.switch[1] = False   
    
    #save data to CSV
    dfa.to_csv('../Data/' + name + '_cleaned.csv')
    
    return (taskinfo, df,dfa)

