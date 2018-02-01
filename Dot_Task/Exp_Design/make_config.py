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
import yaml
  
class Config(object):
    def __init__(self, subjid, taskname, action_keys,
                 stim_repetitions, exp_len, 
                 distribution=norm, seed=None):
        self.subjid = subjid
        self.taskname = taskname
        self.stim_repetitions = stim_repetitions
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.exp_len = exp_len
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.loc = '../Config_Files/'
        self.action_keys = action_keys
        # set up distributions
        if distribution:
            self.distribution = distribution
            try:
                self.distribution_name = distribution.name
            except AttributeError:
                self.distribution_name = 'unknown'
        # set up generic task variables
        self.trial_list = None
        self.base_speed = .1
        self.stim_motions = ['in','out']
        self.stim_oris = [-60,30]
    
    def get_config(self, save=True, filey=None, other_params={}):
        if self.trial_list==None:
            self.setup_trial_list()
        
        initial_params = {
                'subjid': self.subjid,
                'taskname': self.taskname,
                'action_keys': self.action_keys,
                'trigger_key': '5',
                'quit_key': 'q',
                'exp_len': self.exp_len,
                'stim_oris': self.stim_oris,
                'stim_motions': self.stim_motions,
                'base_speed': self.base_speed,
                'clearAfterResponse': 1,
                'responseWindow': 1.0}
                
        initial_params.update(other_params)
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
        for k,v in configuration.items():
            self.__dict__[k] = v
        for k,v in kwargs.items():
            self.__dict__[k] = v  
        # setup
        self.setup_stims()
        
    def setup_stims(self):
        stim_ids = []
        for direction, motion_difficulty in self.motion_difficulties.keys():
            for base_ori, ori_difficulty in self.ori_difficulties.keys():
                for ori_direction in [-1,1]:
                    for speed_direction in [-1,1]:
                        stim_ids.append({'motionDirection': direction,
                                         'oriBase': base_ori,
                                         'speedStrength': motion_difficulty,
                                         'speedDirection': speed_direction,
                                         'oriStrength': ori_difficulty,
                                         'oriDirection': ori_direction})

        self.stim_ids = stim_ids
        
        
class ProbContextConfig(Config): 
    def __init__(self, subjid, taskname, ori_difficulties, motion_difficulties,
                 action_keys=None, stim_repetitions=5,
                 exp_len=None, distribution=norm, dist_args=None, seed=None, 
                 ts_order=None, rp=.9):
        
        if not action_keys:
            action_keys = ['down','up','z', 'x']
        # init Base Exp
        super(ProbContextConfig, self).__init__(subjid,
                                                taskname,
                                                action_keys,
                                                stim_repetitions,
                                                exp_len,
                                                distribution,
                                                seed)
        
        self.rp = rp # recursive probability
        self.dist_args = dist_args
        if dist_args == None:
            self.dist_args = [{'loc': -.3, 'scale': .37}, {'loc': .3, 'scale': .37}]
            
        self.ts_order = ts_order
        if ts_order == None:
            self.ts_order = ['motion','orientation']
            r.shuffle(self.ts_order)
        else:
            assert (set(['motion','orientation']) == set(self.ts_order)), \
                'Tasksets not recognized. Must be "motion" and "orientation"'
        self.states = None
        self.trial_states = None
        # stim difficulties
        self.ori_difficulties = ori_difficulties
        self.motion_difficulties = motion_difficulties
        # calculate exp len
        num_stims = len(self.ori_difficulties)*len(self.motion_difficulties)\
                    *len(self.stim_oris)*len(self.stim_motions)
        if exp_len is None:
            self.exp_len = int(stim_repetitions*num_stims)
        else:
            assert exp_len < int(stim_repetitions*num_stims)
            self.exp_len=exp_len
        # setup
        self.setup_stims()
        
    def get_config(self, save=True, filey=None):
        other_params = {'rp': self.rp,
                        'states': self.states,
                        'stim_ids': self.stim_ids,
                        'ts_order': self.ts_order,
                        'ori_difficulties': self.ori_difficulties,
                        'motion_difficulties': self.motion_difficulties}
        return super(ProbContextConfig, self).get_config(save, 
                                                         filey, 
                                                         other_params)
      
    def load_config_settings(self, filename, **kwargs):
        super(ProbContextConfig, self).load_config_settings(filename, **kwargs)
        self.setup_trial_states()
    
    def setup_stims(self):
        super(ProbContextConfig, self).setup_stims()
        self.states = {i: {'ts': self.ts_order[i], 'dist_args': self.dist_args[i]} for i in range(len(self.ts_order))}
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
            
    def setup_trial_list(self, cueDuration=1.5, CSI=.5, stimulusDuration=2, 
                         responseWindow=1, FBDuration=.5, FBonset=.5, 
                         base_ITI=1, displayFB = True):
        if self.seed is not None:
            np.random.seed(self.seed)
        trial_list = []    
        trial_count = 1
        curr_onset = 2 #initial onset time
        stims = r.sample(self.stim_ids*self.stim_repetitions,self.exp_len)   
            
        #define bins. Will set context to center point of each bin
        bin_boundaries = np.linspace(-1,1,11)
        
        for trial in range(self.exp_len):
            # define ITI
            ITI = base_ITI + r.random()*.5
            state = self.states[self.trial_states[trial]]
            dist = self.distribution(**state['dist_args'])
            binned = -1.1 + np.digitize([dist.rvs()],bin_boundaries)*.2
            context_sample = round(max(-1, min(1, binned[0])),2)
            trial_dict = {
                'trial_count': trial_count,
                'state': self.trial_states[trial],
                'ts': state['ts'],
                'task_distributions': self.distribution_name,
                'distribution_args': state['dist_args'],
                'context': context_sample,
                'stim': stims[trial],
                'onset': curr_onset,
                'cueDuration': cueDuration,
                'stimulusDuration': stimulusDuration,
                'FBDuration': FBDuration,
                'FBonset': FBonset,
                'displayFB': displayFB,
                'CSI': CSI,
                'ITI': ITI,
                #option to change based on state and stim
                'reward_amount': 1,
                'punishment_amount': 0
            }

            trial_list += [trial_dict]

            trial_count += 1
            curr_onset += cueDuration+CSI+stimulusDuration+responseWindow\
                            +FBDuration+FBonset+ITI
        self.trial_list = trial_list
       

class ThresholdConfig(Config): 
    def __init__(self, subjid, taskname, action_keys=None, stim_repetitions=5,
                 seed=None, exp_len=None, ts='motion'):             
        
        if not action_keys:
            action_keys = ['down','up','left','right']
        # init Base Exp
        super(ThresholdConfig, self).__init__(subjid,
                                                taskname,
                                                action_keys,
                                                stim_repetitions,
                                                exp_len,
                                                distribution=None,
                                                seed=seed)
        
        # set task set
        assert ts in ['orientation','motion']
        self.ts = ts
        # stim attributes
        # from easy to hard
        # determines the orientation change in degrees
        self.ori_difficulties = {(self.stim_oris[0], 'easy'): 25,
                                 (self.stim_oris[1], 'easy'): 25,
                                 (self.stim_oris[0], 'hard'): 15,
                                 (self.stim_oris[1], 'hard'): 15}
        # motion speeds
        self.motion_difficulties = {(self.stim_motions[0], 'easy'): .04,
                                 (self.stim_motions[1], 'easy'): .04,
                                 (self.stim_motions[0], 'hard'): .02,
                                 (self.stim_motions[1], 'hard'): .02}
        # calculate exp len
        num_stims = len(self.ori_difficulties)*len(self.motion_difficulties)\
                    *len(self.stim_oris)*len(self.stim_motions)*4
        if exp_len is None:
            self.exp_len = int(stim_repetitions*num_stims)
        else:
            assert exp_len < int(stim_repetitions*num_stims)
            self.exp_len=exp_len
        # setup
        self.setup_stims()
    
        
    def get_config(self, save=True, filey=None):
        other_params = {'stim_ids': self.stim_ids,
                        'ts': self.ts,
                        'ori_difficulties': self.ori_difficulties,
                        'motion_difficulties': self.motion_difficulties}
        return super(ThresholdConfig, self).get_config(save, 
                                                       filey, 
                                                       other_params)
        
                    
    def setup_trial_list(self, stimulusDuration=2, responseWindow=1,
                         FBDuration=.5, FBonset=.5, base_ITI=1, 
                         displayFB = True):
        if self.seed is not None:
            np.random.seed(self.seed)
        trial_list = []    
        trial_count = 1
        curr_onset = 2 #initial onset time
        stims = r.sample(self.stim_ids*self.stim_repetitions,self.exp_len)   
        for trial in range(self.exp_len):
            # set ITI
            ITI = base_ITI + r.random()
            trial_dict = {
                'trial_count': trial_count,
                'ts': self.ts,
                'stim': stims[trial],
                'onset': curr_onset,
                'stimulusDuration': stimulusDuration,
                'responseWindow': responseWindow,
                'FBDuration': FBDuration,
                'FBonset': FBonset,
                'displayFB': displayFB,
                'ITI': ITI,
                #option to change based on state and stim
                'reward_amount': 1,
                'punishment_amount': 0
            }

            trial_list += [trial_dict]

            trial_count += 1
            curr_onset += stimulusDuration+responseWindow\
                          +FBDuration+FBonset+ITI
        self.trial_list = trial_list
       