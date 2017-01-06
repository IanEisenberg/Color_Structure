# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:13:45 2014

@author: admin
"""

import datetime
import numpy as np
from os import path
import random as r
from scipy.stats import norm

class ConfigList(object):
    def __init__(self, taskname='taskname', subjid='000', rp=.9,
                 action_keys=None, distribution=norm, exp_len=200, seed=None):
        self.taskname = taskname
        self.subjid = subjid
        self.rp = rp # recursive probability
        self.exp_len = exp_len
        self.distribution = norm
        try:
            self.distribution_name = distribution.name
        except AttributeError:
            self.distribution_name = 'unknown'
        self.seed = seed
        if action_keys == None:
            self.action_keys = ['d','f','j', 'k']
            r.shuffle(self.action_keys)
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.loc = '../Config_Files/'
        self.states = None
        self.trial_states = None
        self.trial_list = None
        if self.seed is not None:
            np.random.seed(self.seed)
        
    def setup_stims(self, ts_order=None, args=None):
        self.stim_ids = [(0,2),(0,3),(1,2),(1,3)]
        if ts_order == None:
            ts_order = [0,1]
            r.shuffle(ts_order)
        if args==None:
            args = [{'loc': -.3, 'scale': .37}, {'loc': -.3, 'scale': .37}]
        self.states = {i: {'ts': ts_order[i], 'dist_args': args[i]} for i in range(len(ts_order))}
        self.setup_trial_states()
        
    def get_config(self, save=True, filey=None):
        if self.states==None:
            self.setup_stims()
            self.setup_trial_states()
        if self.trial_list==None:
            self.setup_trial_list()
        
        initial_params = {
          'clearAfterResponse': 1,
          'quit_key': 'q',
          'responseWindow': 1.0,
          'taskname': self.taskname,
          'id': self.subjid,
          'trigger_key': '5',
          'action_keys': self.action_keys,
          'states': self.states,
          'rp': self.rp,
          'exp_len': self.exp_len,
          'stim_ids': self.stim_ids
        }
        
        to_save = self.trial_list
        to_save.insert(0,initial_params)
        if save==True:
            filename = self.taskname + '_' + self.subjid + '_config_' + self.timestamp + '.npy'
            if filey == None:
                filey = path.join(self.loc,filename)
            np.save(filey, to_save)
            return filey
        else:
            return to_save
    
    def setup_trial_states(self):
        """
        Create a list of trials with the correct block length. Define tasksets with
        "probs" = P(reward | correct) and P(reward | incorrect), and "actions" =
        correct action for stim 1 and stim 2.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        trans_probs = np.matrix([[self.rp, 1-self.rp], [1-self.rp, self.rp]])
        trial_states = [1] #start off the function
        #creates the task-set trial list. Task-sets alternate based on recusive_p
        #with a maximum repetition of 25 trials. This function also makes sure
        #that each task-set composes at least 40% of trials
        while abs(np.mean(trial_states)-.5) > .1:
            curr_state = r.choice(list(self.states.keys()))
            trial_states = []
            state_reps = 0
            for trial in range(self.exp_len):
                trial_states.append(curr_state)
                if r.random() > trans_probs[curr_state,curr_state] or state_reps > 25:
                    curr_state = 1-curr_state
                    state_reps = 0
                else:
                    state_reps += 1
        self.trial_states = trial_states
            
                    
    def setup_trial_list(self, stimulusDuration=1.5, FBDuration=.5, FBonset=.5, ITI=.5):
        if self.seed is not None:
            np.random.seed(self.seed)
        trial_list = []    
        trial_count = 1
        curr_onset = 2 #initial onset time
        stims = r.sample(self.stim_ids*int(self.exp_len/4.0),self.exp_len)   
            
        #define bins. Will set context to center point of each bin
        bin_boundaries = np.linspace(-1,1,11)
        
        
        for trial in range(self.exp_len):
            state = self.states[self.trial_states[trial]]
            dist = self.distribution(**state['dist_args'])
            binned = -1.1 + np.digitize([dist.rvs()],bin_boundaries)*.2
            context_sample = round(max(-1, min(1, binned[0])),2)

            
            trial_list += [{
                'trial_count': trial_count,
                'state': self.trial_states[trial],
                'ts': state['ts'],
                'task_distributions': self.distribution_name,
                'distribution_args': state['dist_args'],
                'context': context_sample,
                'stim': stims[trial],
                'onset': curr_onset,
                'FBDuration': FBDuration,
                'FBonset': FBonset,
                'StimulusDuration': stimulusDuration,
                'ITI': ITI,
                #option to change based on state and stim
                'reward': 1,
                'punishment': 0
            }]

            trial_count += 1
            curr_onset += stimulusDuration+FBDuration+FBonset+ITI+r.random()*.5
        self.trial_list = trial_list
       

