# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:13:45 2014

@author: admin
"""

import numpy as np
import random as r
import yaml
import datetime

def makeConfigList(taskname = 'Temp_Struct', iden = '000', probs1 = (.8, .2),
                    probs2 = (.8, .2), num_blocks = 50, block_len = 16, 
                    action_keys = None, loc = '../Config_Files/'):
    
    timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    iden = str(iden)
    if not action_keys:
        action_keys = ['h', 'j', 'k', 'l']
        r.shuffle(action_keys)
    stim_ids = [0,1]
    tasksets = {'ts1': {'probs': probs1, 'actions': [action_keys[i] for i in (0,1)]}, 
                          'ts2': {'probs': probs2, 'actions': [action_keys[i] for i in (2,3)]}}
    initial_params = {
      'clearAfterResponse': 1,
      'quit_key': 'q',
      'responseWindow': 1.0,
      'stimulusDuration': 1.0,  
      'FBDuration': 1.0,
      'taskname': taskname,
      'id': iden,
      'trigger_key': '5',
      'action_keys': action_keys,
      'tasksets': tasksets,
      'stim_ids': stim_ids,
      'block_len': block_len
    }
    
    def makeTrialList():
        """
        Create a list of trials with the correct block length. Define tasksets with
        "probs" = P(reward | correct) and P(reward | incorrect), and "actions" =
        correct action for stim 1 and stim 2.
        """
        trialList = []    
        keys = tasksets.keys()
        r.shuffle(keys)
        trial_count = 1
        curr_onset = 1 #initial onset
       
       
       #ensure that the valid trial probabilities for each taskset are close to
       #the desired probabilities
       
        diff = np.array([1,1])   
        ts1_probs = np.array(tasksets['ts1']['probs'])
        while (diff > [.025,.025]).any():
                ts1_valid = [np.random.binomial(1, ts1_probs[0],block_len*num_blocks/2),np.random.binomial(1, ts1_probs[1],block_len*num_blocks/2)]
                ts1_valid_probs = np.array([np.mean(array) for array in ts1_valid])
                diff = np.array(abs(ts1_probs - ts1_valid_probs))
        
        diff = np.array([1,1])    
        ts2_probs = np.array(tasksets['ts2']['probs'])
        while (diff > [.025,.025]).any():
                ts2_valid = [np.random.binomial(1, ts2_probs[0],block_len*num_blocks/2),np.random.binomial(1, ts2_probs[1],block_len*num_blocks/2)]
                ts2_valid_probs = np.array([np.mean(array) for array in ts2_valid])
                diff = np.array(abs(ts2_probs - ts2_valid_probs))

        valid_trials = [ts1_valid, ts2_valid]
            
            
        #create blocks with a random order of stims (keeping the freqs equal)
        for block in range(num_blocks):
            #Define stim order, ensuring that no stim appears more than 3 times 
            #in a row and the overall proportion of stim1 is close to 50%
            mean_stim = 0
            local_var = []
            while (abs(mean_stim - .5) >= .1) or (0 in local_var):
                stims = r.sample(stim_ids*int(block_len*.8),block_len)
                mean_stim = np.mean(stims)
                local_var = [np.var(stims[i:i+4]) for i in range(block_len-4)]
            #Alternate tasksets
            curr_ts = tasksets[keys[block%2]]
            
        
            PosFB_c = valid_trials[block%2][0][0:12]
            PosFB_i = valid_trials[block%2][1][0:12]
            valid_trials[block%2][0] = valid_trials[block%2][0][12:]
            valid_trials[block%2][1] = valid_trials[block%2][1][12:]

            

            for trial in range(block_len):
                trialList += [{
                    'trial_count': trial_count,
                    'TS': curr_ts,
                    'stim': stims[trial],
                    'correct_action': curr_ts['actions'][stims[trial]],
                    'onset': curr_onset,
                    'FBonset': .5,
                    'PosFB_correct': bool(PosFB_c[trial]),
                    'PosFB_incorrect': bool(PosFB_i[trial])
                }]
                trial_count += 1
                curr_onset += 2.5+r.random()*.5
                
        return trialList

    
    yaml_input = makeTrialList()
    yaml_input.insert(0,initial_params)    
    filename = taskname + '_' + iden + '_config_' + timestamp + '.yaml'
    f=open(loc + filename,'w')
    yaml.dump_all(yaml_input,f,default_flow_style = False, explicit_start = True)
    return loc+filename
    