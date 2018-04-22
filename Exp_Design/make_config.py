# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:13:45 2014

@author: admin
"""
from copy import deepcopy
import datetime
from glob import glob
import numpy as np
import os
from os import path
import random as r
from scipy.stats import norm
import yaml
file_dir = os.path.dirname(__file__)

# helper function
def split_config(config, trials_per_run):
    config_files = []
    total_trials = len(config.trial_list)-1
    start = 1
    run=1
    while start < total_trials:
        new_config = deepcopy(config)
        # subset config
        new_config.trial_list = new_config.trial_list[start:start+trials_per_run]
        start += trials_per_run
        config_files.append(new_config.get_config(setup_args={'displayFB': False},
                                                  run=run))
        run+=1
    return config_files

# Class
class Config(object):
    def __init__(self, subjid, taskname, action_keys,
                 stim_repetitions, distribution=norm, seed=None):
        self.subjid = subjid
        self.taskname = taskname
        self.stim_repetitions = stim_repetitions
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.loc = path.join(file_dir, '..','Config_Files',subjid)
        # make loc if not made
        try:
            os.makedirs(self.loc)
        except OSError:
            pass
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
        self.base_speed = .12
        self.stim_motions = ['in','out']
        self.stim_oris = [-45,45]
    
    def get_config(self, save=True, filey=None, run=None,
                   other_params=None, setup_args=None):
        if other_params is None:
            other_params = {}
        if setup_args is None:
            setup_args = {}
        if self.trial_list==None:
            self.setup_trial_list(**setup_args)
        
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
            basefilename = self.taskname + '_' + self.subjid + '_config_' + self.timestamp 
            if run is None:
                similar_files = glob(path.join(self.loc,basefilename+'*'))
                filename = basefilename + '_Run%s.yaml' % str(len(similar_files)+1).zfill(2)
            else:
                filename = basefilename + '_Run%s.yaml' % str(run).zfill(2)
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
        for direction, speed_difficulty in self.speed_difficulties.keys():
            for base_ori, ori_difficulty in self.ori_difficulties.keys():
                for ori_direction in [-1,1]:
                    for speed_direction in [-1,1]:
                        stim_ids.append({'motionDirection': direction,
                                         'oriBase': base_ori,
                                         'speedStrength': speed_difficulty,
                                         'speedDirection': speed_direction,
                                         'oriStrength': ori_difficulty,
                                         'oriDirection': ori_direction})

        self.stim_ids = stim_ids
        
        
class ProbContextConfig(Config): 
    def __init__(self, subjid, taskname, 
                 ori_difficulties, speed_difficulties, responseCurves,
                 action_keys=None, stim_repetitions=5,
                 exp_len=None, distribution=norm, dist_args=None, seed=None, 
                 ts_order=None, rp=.9):
        
        if not action_keys:
            action_keys = ['down','up','left','right']
        # init Base Exp
        super(ProbContextConfig, self).__init__(subjid,
                                                taskname,
                                                action_keys,
                                                stim_repetitions,
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
        self.ori_difficulties = {}
        for difficulty in ori_difficulties:
            for pedestal in self.stim_oris:
                val = responseCurves['orientation'].inverse(difficulty)
                self.ori_difficulties[(pedestal, difficulty)] = val
        self.speed_difficulties = {}
        for difficulty in speed_difficulties:
            for pedestal in self.stim_motions:
                val = responseCurves['motion'].inverse(difficulty)
                self.speed_difficulties[(pedestal, difficulty)] = val
        # calculate exp len
        num_stims = len(self.ori_difficulties)*len(self.speed_difficulties)\
                    *len(self.stim_oris)*len(self.stim_motions)
        if exp_len is None:
            self.exp_len = int(stim_repetitions*num_stims)
        else:
            assert exp_len < int(stim_repetitions*num_stims)
            self.exp_len=exp_len
        # setup
        self.setup_stims()
        
    def get_config(self, save=True, filey=None, setup_args=None, run=None):
        other_params = {'rp': self.rp,
                        'states': self.states,
                        'stim_ids': self.stim_ids,
                        'ts_order': self.ts_order,
                        'ori_difficulties': self.ori_difficulties,
                        'speed_difficulties': self.speed_difficulties}
        return super(ProbContextConfig, self).get_config(save, 
                                                         filey, 
                                                         run,
                                                         other_params,
                                                         setup_args)
      
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
            
    def setup_trial_list(self, cueDuration=.75, CSI=.25, 
                         stimulusDuration=1.5, responseWindow=1, avg_SRI=2,
                         FBDuration=1, FBonset=0, avg_ITI=4, displayFB=True,
                         counterbalance_task=False):
        if self.seed is not None:
            np.random.seed(self.seed)
        if counterbalance_task:
            self.exp_len = self.exp_len*2
        curr_onset = 2 #initial onset time
        trial_list = [] 
        # define ITI and SRI
        ITIs = []; SRIs = []; 
        for _ in range(self.exp_len):
            ITIs.append(min(np.random.exponential(avg_ITI-1)+1, 3*avg_ITI))
            SRIs.append(min(np.random.exponential(avg_SRI-.5)+.5, 3*avg_SRI))
                
        if not counterbalance_task:
            stims = r.sample(self.stim_ids*self.stim_repetitions,self.exp_len)   
            #define bins. Will set context to center point of each bin
            bin_boundaries = np.linspace(-1,1,11)
            for trial in range(self.exp_len):
                state = self.states[self.trial_states[trial]]
                dist = self.distribution(**state['dist_args'])
                binned = -1.1 + np.digitize([dist.rvs()],bin_boundaries)*.2
                context_sample = round(max(-1, min(1, binned[0])),2)
                trial_dict = {
                    'trial_count': trial+1,
                    'state': self.trial_states[trial],
                    'ts': state['ts'],
                    'task_distributions': self.distribution_name,
                    'distribution_args': state['dist_args'],
                    'context': context_sample,
                    'stim': stims[trial].copy(),
                    'onset': curr_onset,
                    'cueDuration': cueDuration,
                    'stimulusDuration': stimulusDuration,
                    'responseWindow': responseWindow,
                    'stimResponseInterval': SRIs[trial],
                    'FBDuration': FBDuration,
                    'FBonset': FBonset,
                    'displayFB': displayFB,
                    'CSI': CSI,
                    'ITI': ITIs[trial],
                    #option to change based on state and stim
                    'reward_amount': 1,
                    'punishment_amount': 0
                }
                trial_list += [trial_dict]
                curr_onset += cueDuration+CSI+stimulusDuration+responseWindow\
                                +SRIs[trial]+FBDuration+FBonset+ITIs[trial]
        else:
            stims = self.stim_ids*self.stim_repetitions*2
            trial=0
            for _ in range(self.exp_len//2):
                for task in self.ts_order:
                    trial_dict = {
                        'trial_count': trial+1,
                        'ts': task,
                        'stim': stims[trial].copy(),
                        'onset': curr_onset,
                        'cueDuration': cueDuration,
                        'stimulusDuration': stimulusDuration,
                        'responseWindow': responseWindow,
                        'stimResponseInterval': SRIs[trial],
                        'FBDuration': FBDuration,
                        'FBonset': FBonset,
                        'displayFB': displayFB,
                        'CSI': CSI,
                        'ITI': ITIs[trial],
                        #option to change based on state and stim
                        'reward_amount': 1,
                        'punishment_amount': 0
                    }
                    trial_list += [trial_dict]
                    curr_onset += cueDuration+CSI+stimulusDuration+responseWindow\
                                    +SRIs[trial]+FBDuration+FBonset+ITIs[trial]
                    trial += 1
        self.trial_list = trial_list
       

class ThresholdConfig(Config): 
    def __init__(self, subjid, taskname, action_keys=None, stim_repetitions=5,
                 seed=None, exp_len=None, ts='motion', one_difficulty=False):             
        
        if not action_keys:
            action_keys = ['down','up','left','right']
        # init Base Exp
        super(ThresholdConfig, self).__init__(subjid,
                                                taskname,
                                                action_keys,
                                                stim_repetitions,
                                                distribution=None,
                                                seed=seed)
        
        # set task set
        assert ts in ['orientation','motion']
        self.ts = ts
        # stim attributes
        if not one_difficulty:
        # from easy to hard
            # determines the orientation change in degrees
            self.ori_difficulties = {(self.stim_oris[0], .85): 30,
                                     (self.stim_oris[1], .85): 30,
                                     (self.stim_oris[0], .7): 20,
                                     (self.stim_oris[1], .7): 20}
            # motion speeds
            self.speed_difficulties = {(self.stim_motions[0], .85): .08,
                                     (self.stim_motions[1], .85): .08,
                                     (self.stim_motions[0], .7): .04,
                                     (self.stim_motions[1], .7): .04}
        else:
            self.ori_difficulties = {(self.stim_oris[0], .775): 30,
                                     (self.stim_oris[1], .775): 30}
            # motion speeds
            self.speed_difficulties = {(self.stim_motions[0], .775): .06,
                                        (self.stim_motions[1], .775): .06}
            
        # calculate exp len
        num_stims = len(self.ori_difficulties)*len(self.speed_difficulties)\
                    *len(self.stim_oris)*len(self.stim_motions)
        if exp_len is None:
            self.exp_len = int(stim_repetitions*num_stims)
        else:
            assert exp_len < int(stim_repetitions*num_stims)
            self.exp_len=exp_len
        # setup
        self.setup_stims()
    
        
    def get_config(self, save=True, filey=None, run=None, setup_args=None):
        other_params = {'stim_ids': self.stim_ids,
                        'ts': self.ts,
                        'ori_difficulties': self.ori_difficulties,
                        'speed_difficulties': self.speed_difficulties}
        return super(ThresholdConfig, self).get_config(save, 
                                                       filey, 
                                                       run,
                                                       other_params,
                                                       setup_args)
        
                    
    def setup_trial_list(self, stimulusDuration=1.5, responseWindow=1.5, 
                         avg_SRI=.5, FBDuration=.5, FBonset=0, avg_ITI=.5, 
                         displayFB = True):
        if self.seed is not None:
            np.random.seed(self.seed)
        trial_list = []    
        trial_count = 1
        curr_onset = 2 #initial onset time
        stims = r.sample(self.stim_ids*self.stim_repetitions,self.exp_len)   
        for trial in range(self.exp_len):
            # define ITI and SRI
            ITI = avg_ITI + r.random()-.5
            SRI = avg_SRI + r.random()-.5
            trial_dict = {
                'trial_count': trial_count,
                'ts': self.ts,
                'stim': stims[trial].copy(),
                'onset': curr_onset,
                'stimulusDuration': stimulusDuration,
                'responseWindow': responseWindow,
                'stimResponseInterval': SRI,
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
                          +SRI+FBDuration+FBonset+ITI
        self.trial_list = trial_list
       