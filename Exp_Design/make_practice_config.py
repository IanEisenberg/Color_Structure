# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:13:45 2014

@author: admin
"""

import numpy as np
import random as r
import yaml

def makeConfigList(taskname = 'Temp_Struct', loc = '../Config_Files/'):
    
    probs1 = probs2 = (1,0)
    stim_ids = [0,1]
    block_len = 15
    action_keys = ['h', 'j', 'k', 'l']
    tasksets = {'ts1': {'probs': probs1, 'actions': [action_keys[i] for i in (0,1)]}, 
                          'ts2': {'probs': probs2, 'actions': [action_keys[i] for i in (2,3)]}}
    initial_params = {
      'clearAfterResponse': 1,
      'quit_key': 'q',
      'responseWindow': 1.0,
      'stimulusDuration': 1.0,
      'FBDuration': 1.0,
      'taskname': taskname,
      'trigger_key': '5',
      'action_keys': action_keys,
      'tasksets': tasksets,
      'stim_ids': stim_ids,
      'block_len': block_len
    }
    
    def makeTrialList(block_len):
        """
        Create a list of trials with the correct block length. Define tasksets with
        "probs" = P(reward | correct) and P(reward | incorrect), and "actions" =
        correct action for stim 1 and stim 2.
        """
        trialList = []    
        curr_onset = 1 #initial onset
        #Create practice stim order
        r.seed(11111)
        stims = r.sample(stim_ids*int(block_len*.8),block_len)

        #Select Taskset for practice
        curr_ts = tasksets['ts1']
        
        PosFB_c = [1]*block_len
        PosFB_i = [0]*block_len
        PosFB_c[11] = 0
        PosFB_i[11] = 1

        

        for trial in range(block_len):
            trialList += [{
                'TS': curr_ts,
                'stim': stims[trial],
                'correct_action': curr_ts['actions'][stims[trial]],
                'onset': curr_onset,
                'FBonset': .5,
                'PosFB_correct': bool(PosFB_c[trial]),
                'PosFB_incorrect': bool(PosFB_i[trial])
            }]
            curr_onset += 3+r.random()
                
        return trialList
    
    yaml_input = makeTrialList(block_len)
    yaml_input.insert(0,initial_params)    
    filename = 'Temp_Struct_Practice_config.yaml'
    f=open(loc + filename,'w')
    yaml.dump_all(yaml_input,f,default_flow_style = False, explicit_start = True)

makeConfigList()    