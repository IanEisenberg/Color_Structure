# -*- coding: utf-8 -*-
"""
Define a bot to run temp_struct_task
"""

import numpy as np
from scipy.stats import norm
import yaml

class test_bot:
    """class defining a test bot based on a psych task
    """
    
    def __init__(self, config_file, dist=norm, mode = "one-shot"):
        config_file = yaml.load(open(config_file,'r'))
        self.taskinfo = config_file[0]
        for k in self.taskinfo.iterkeys():
                    self.__dict__[k]=self.taskinfo[k]
        #shadegenerators for each state
        self.state_dis = [dist(**state['dist_args']) for state in self.taskinfo['states'].values()] 
        self.TS_prior = [.5,.5]
        self.posterior = [.5, .5]
        self.mode = mode
                    
                    
    
    def choose(self, stim, context):
        """choose an action based on Qvalues and get a RT for choice
        """
        self.updateLikelihood(context)
        curr_state = self.taskinfo['states'][np.argmax(self.posterior)]
        choice = self.action_keys[stim[curr_state['ts']]]
        return (choice,.5)

        
    def updateLikelihood(self, context):
        support = [dis.pdf(context) for dis in self.state_dis]
        self.posterior = np.array([support[i]*self.TS_prior[i] for i in range(len(support))])/ \
                    np.sum([support[i]*self.TS_prior[i] for i in range(len(support))])
        if self.mode == "one-shot":
            self.TS_prior[np.argmax(self.posterior)] = self.recursive_p
            self.TS_prior[1-np.argmax(self.posterior)] = 1-self.recursive_p
        elif self.mode == "ignore_base":
            self.TS_prior = [.5, .5]
        else:
            self.TS_prior = [sum(self.posterior*[self.recursive_p,1-self.recursive_p]), 
                             sum(self.posterior*[1-self.recursive_p,self.recursive_p])]
        
        
    

