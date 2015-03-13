# -*- coding: utf-8 -*-
"""
Define a bot to run temp_struct_task
"""

import yaml
import numpy as np
import random as r

class test_bot:
    """class defining a test bot based on a psych task
    """
    
    def __init__(self, config_file):
        config_file = yaml.load_all(file(config_file))
        self.taskinfo = config_file.next()
        for k in self.taskinfo.iterkeys():
                    self.__dict__[k]=self.taskinfo[k]
        self.Q = Qlearning(self.action_keys,self.stim_ids)
    
    def choose(self, stim):
        """choose an action based on Qvalues and get a RT for choice
        """
        choice = self.Q.getAction_SM(stim)
        RT = self.Q.getRT()
        return (choice,RT)
        
    def learn(self, reward):
        """update Q values from reward
        """
        self.Q.updateQ(reward)
        
    

        
     
class Qlearning:
    """class that takes states and actions and learns Q-value for each
    state-action pair
    """
    
    def __init__(self,actions, states):
        self.actions = actions
        self.states = states
        self.last_action=[]
        self.last_state=[]
        self.Qstates = {self.states[i]:[10.0]*len(self.actions) for i in range(len(self.states))}
        self.Qstate_num = {self.states[i]:[0]*len(self.actions) for i in range(len(self.states))}
        
    def getQValue(self,action,state):
        """ get Q(s,a) as well as the number of times (s,a) has been seen
        """
        action_in = self.actions.index(action)
        return (self.Qstates[state][action_in], self.Qstate_num[state][action_in])
    
    def setQValue(self,action,state, value,num):
        """ set Q(s,a)
        """
        action_in = self.actions.index(action)
        self.Qstates[state][action_in] = value
        self.Qstate_num[state][action_in] = num
        
    def updateQ(self, reward):
        """
        If (s,a) has never been seen, set Q(s,a) to the value of the rewards. 
        If (s,a) has been seen, update Q(s,a) according to the Q-learning rule,
        inveresely weighting new experiences by the number of times (s,a) has
        been seen.
        """
        val,num = self.getQValue(self.last_action,self.last_state)
        if num == 0:
            self.setQValue(self.last_action,self.last_state,reward,1)
        else:
            curr_Q = val
            alpha = 1.0/num
            update = curr_Q + alpha * (reward - curr_Q)
            self.setQValue(self.last_action,self.last_state,update,num+1) 
            
    def getRT(self):
        """
        Unsunstantiated RT rule. Here RT is just a random distribution around .5
        """
        RT = r.normalvariate(.5,.1) 
        if RT < .1:
            RT = .1
        return RT
    
    def getAction(self,state):
        """
        If state s has never been seen, choose a random action. If s has been
        seen, choose the action with the highest Q value. If multiple actions
        have the same Q value, randomly choose one. There is an exploration 
        parameter such that a random action will be chosen a percent
        of the time proportional to the number of times state s had been seen.
        Thus exploration goes down over time.
        """
        self.last_state = state
        num_exposures = sum(self.Qstate_num[state])
        if num_exposures == 0:
            self.last_action = r.choice(self.actions)
        else:
            if r.random() < max(.15,(1.0/num_exposures**.5)):
                self.last_action = r.choice(self.actions)
            else:
                maxQ = max(self.Qstates[state])
                max_index = [index for index,value in enumerate(self.Qstates[state]) 
                            if value == maxQ]
                self.last_action = self.actions[r.choice(max_index)]
        return self.last_action
            
    def getAction_SM(self,state):
        """
        Chooses action using the softmax function
        """
        self.last_state = state
        num_exposures = sum(self.Qstate_num[state])
        T = max(1-(num_exposures)*.01,.1)
        Qstates = np.array(self.Qstates[state])
        expQ = np.exp(Qstates/T)
        softmax = expQ/sum(expQ)
        self.last_action = np.random.choice(self.actions,1,p = softmax)[0]
        return self.last_action
                
        
            
            
        
            
        
        
    
