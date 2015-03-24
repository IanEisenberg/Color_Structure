# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:13:45 2014

@author: admin
"""

import numpy as np
from scipy.stats import norm
import random as r
import yaml
import datetime

def makeConfigList(taskname = 'Color_Struct', iden = '000', 
                   recursive_p = .9, 
                   ts1 = [[1,0,0,0],[0,1,0,0]],
                   ts2 = [[0,0,0,0],[0,0,0,1]],
                   exp_len = 200,
                    action_keys = None, loc = '../Config_Files/'):
    
    trans_probs = np.matrix([[recursive_p, 1-recursive_p], [1-recursive_p, recursive_p]])
    timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    iden = str(iden)
    if not action_keys:
        action_keys = ['h', 'j', 'k', 'l']
        r.shuffle(action_keys)
    stim_ids = [0,1]
    #each taskset is define as a nxm matrix where n = # of stims and
    #m = # of actions. In theory, 'n' could be further decomposed into features
    states = {0: {'ts': ts1, 'c_mean': -.3, 'c_sd': .37}, 
                1: {'ts': ts2, 'c_mean': .3, 'c_sd': .37}}

    #useful if I wanted to parametrically alter overlap
#    def minf1f2(x, mu1, mu2, sd1, sd2):
#        f1 = norm(mu1, sd1).pdf(x)
#        f2 = norm(mu2, sd2).pdf(x)
#        return min(f1, f2)
#    overlap = scipy.integrate.quad(minf1f2,-np.Inf,np.Inf,args = (-.4, .4, .5, .5))

                
    initial_params = {
      'clearAfterResponse': 1,
      'quit_key': 'q',
      'responseWindow': 1.0,
      'stimulusDuration': 1.0,  
      'FBDuration': .5,
      'taskname': taskname,
      'id': iden,
      'trigger_key': '5',
      'action_keys': action_keys,
      'states': states,
      'stim_ids': stim_ids,
      'exp_len': exp_len
    }
    
    
    def makeTrialList():
        """
        Create a list of trials with the correct block length. Define tasksets with
        "probs" = P(reward | correct) and P(reward | incorrect), and "actions" =
        correct action for stim 1 and stim 2.
        """
        trialList = []    
        trial_count = 1
        curr_onset = 1 #initial onset
        curr_state = r.choice(states.keys())
        stims = r.sample(stim_ids*int(exp_len*.5),exp_len)
                
        
        
        for trial in range(exp_len):
            state = states[curr_state]
            dis = norm(state['c_mean'],state['c_sd'])
            context_sample = [max(-1, min(1, dis.rvs()))] * 3

            
            trialList += [{
                'trial_count': trial_count,
                'state': curr_state,
                'ts': state['ts'],
                'c_dis': {'mean': dis.mean(), 'sd': dis.std()},
                'context': context_sample,
                'stim': stims[trial],
                'onset': curr_onset,
                'FBonset': .5,
            }]
            if r.random() > trans_probs[curr_state,curr_state]:
                curr_state = 1-curr_state
            
            trial_count += 1
            curr_onset += 2.5+r.random()*.5
        
       
        
                
        return trialList

    
    np_input = makeTrialList()
    np_input.insert(0,initial_params)
    filename = taskname + '_' + iden + '_config_' + timestamp + '.npy'
    np.save(loc + filename, np_input)
    
#    yaml_input = makeTrialList()
#    yaml_input.insert(0,initial_params)    
#    filename = taskname + '_' + iden + '_config_' + timestamp + '.yaml'
#    f=open(loc + filename,'w')
#    yaml.dump_all(yaml_input,f,default_flow_style = False, explicit_start = True)
    return loc+filename
    