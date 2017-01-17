# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:13:45 2014

@author: admin
"""

import datetime
import yaml
import numpy as np
from os import path
import random as r
from scipy.stats import norm

class ConfigList(object):
    def __init__(self, taskname='taskname', subjid='000', rp=.9,
                 action_keys=None, distribution=norm, args=None, exp_len=200, ts_order=None, seed=None):
        self.seed = seed
        if self.seed is not None:
            r.seed(self.seed)
            np.random.seed(self.seed)
        self.distribution = norm
        self.exp_len = exp_len
        self.rp = rp # recursive probability
        self.subjid = subjid
        self.ts_order = ts_order
        self.taskname = taskname
        try:
            self.distribution_name = distribution.name
        except AttributeError:
            self.distribution_name = 'unknown'
        if action_keys == None:
            self.action_keys = ['d','f','j', 'k']
            r.shuffle(self.action_keys)
        if args == None:
            self.args = [{'loc': -.3, 'scale': .37}, {'loc': .3, 'scale': .37}]
        if ts_order == None:
            self.ts_order = [0,1]
            r.shuffle(self.ts_order)
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.loc = '../Config_Files/'
        self.states = None
        self.trial_states = None
        self.trial_list = None
        
        # setup
        self.setup_stims()
        self.setup_trial_states()
    
    def get_config(self, save=True, filey=None):
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
          'stim_ids': self.stim_ids,
          'ts_order': self.ts_order
        }
        
        to_save = self.trial_list
        to_save.insert(0,initial_params)
        if save==True:
            filename = self.taskname + '_' + self.subjid + '_config_' + self.timestamp + '.yaml'
            if filey == None:
                filey = path.join(self.loc,filename)
            yaml.dump(to_save, open(filey,'w'))
            return filey
        else:
            return to_save
      
    def load_config_settings(self, filename, **kwargs):
        if not path.exists(filename):
            raise BaseException('Config file not found')
        config_file = yaml.load(open(filename,'r'))
        configuration = config_file[0]
        self.__dict__.update(configuration)
        self.__dict__.update(kwargs)
        # setup
        self.setup_stims()
        self.setup_trial_states()
        
    def setup_stims(self):
        self.stim_ids = [(0,2),(0,3),(1,2),(1,3)]
        self.states = {i: {'ts': self.ts_order[i], 'dist_args': self.args[i]} for i in range(len(self.ts_order))}
        self.setup_trial_states()
  
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
            
                    
    def setup_trial_list(self, stimulusDuration=1.5, FBDuration=.5, FBonset=.5, ITI=.5, displayFB = True):
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
                'displayFB': displayFB,
                'StimulusDuration': stimulusDuration,
                'ITI': ITI,
                #option to change based on state and stim
                'reward_amount': 1,
                'punishment_amount': 0
            }]

            trial_count += 1
            curr_onset += stimulusDuration+FBDuration+FBonset+ITI+r.random()*.5
        self.trial_list = trial_list
       

