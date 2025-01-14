# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:09:29 2015

@author: Ian
"""

import random as r
import numpy as np

def softmax(probs, inv_temp):
    return np.exp(probs*inv_temp)/sum(np.exp(probs*inv_temp))
    
class BiasPredModel:
    """
    Prediction model that takes in data, and uses a prior over hypotheses
    and the relevant to calculate posterior hypothesis estimates. Eps
    determines the probability of acting randomly each trial which translates
    into a 'mixture model' calculation for the trial-by-trial posterior
    """
    def __init__(self, likelihood_dist, prior, r1 = .9, r2 = .9, rp = None, TS_eps = 0,
                 action_eps = 0):
        self.prior = np.array(prior)
        self.likelihood_dist = likelihood_dist
        if rp:
            self.r1 = rp
            self.r2 = rp
        else:
            self.r1 = r1
            self.r2 = r2
        self.posterior = prior
        self.TS_eps = TS_eps
        self.action_eps = action_eps
        
    def calc_posterior(self, context):
        """
        Calculate the posterior probability of different distribution hypotheses (TSs)
        given a context point. You can pass in multiple context points but this function
		will only calculate the posterior based on the current prior and will not update
		in between each context point.
        """
        ld = self.likelihood_dist
        r1 = self.r1
        r2 = self.r2
        eps = self.TS_eps
        prior = self.prior
           
        trans_probs = np.array([[r1, 1-r1], [1-r2, r2]]).transpose()            
        n = len(prior)
        likelihood = np.array([dis.pdf(context) for dis in ld])
        numer = np.array([likelihood[i] * prior[i] for i in range(n)])
        dinom = np.sum(numer,0)
        posterior = numer/dinom
        self.prior = np.dot(trans_probs,posterior)
        self.posterior = posterior
        TS_probs = (1-eps)*posterior+eps/2  # mixed model of TS posteriors and random guessing
        return TS_probs
        
    def calc_action_posterior(self, stim, context):
        """
        Calculate the posterior probability of different actions. Pass in one
        context value and one stim
        """
        self.calc_posterior(context)
        TS_eps = self.TS_eps
        eps = self.action_eps
        TS_probs = (1-TS_eps)*self.posterior+TS_eps/2
        action_probs = np.zeros(4)
        for i in range(len(stim)):
            action_probs[stim[i]] = TS_probs[i]
        action_probs = (1-eps)*action_probs+eps/len(action_probs)
        return action_probs
    
       
    def choose(self, mode = 'e-greedy', eps = None, inv_temp = 1):
        if eps == None:
            eps = self.TS_eps
        if mode == "e-greedy":
            TS_probs = (1-eps)*self.posterior+eps/2
            return np.random.choice(range(len(TS_probs)), p = TS_probs)
        elif mode == 'prob_match':
            return np.random.choice(range(len(self.posterior)), p = self.posterior)
        elif mode == "softmax":
            probs = softmax(self.posterior, inv_temp)
            return np.random.choice(range(len(probs)), p = probs)
        else:
            return np.argmax(self.posterior)

class MemoryModel:
    """
    Prediction model that takes in data, and uses a prior over hypotheses
    and the relevant to calculate posterior hypothesis estimates. Eps
    determines the probability of acting randomly each trial which translates
    into a 'mixture model' calculation for the trial-by-trial posterior
    :param k: discount rate over prior contexts
    :param perseverance: bias to responding the same TS as the last trial
    :param bias: bias to responding TS2. Higher values indicate greater P(TS2)
    """
    def __init__(self, likelihood_dist, k=1, perseverance = 0, bias = .5, TS_eps = 0,
                 action_eps = 0):
        self.likelihood_dist = likelihood_dist
        self.k = k
        self.perseverance = perseverance
        self.history= []
        self.TS_probs = []
        self.TS_eps = TS_eps
        self.bias = [1-bias, bias]
        self.action_eps = action_eps
        
    def calc_posterior(self, context, last_TS=None):
        """
        Calculate the posterior probability of different distribution hypotheses (TSs)
        given a context point. You can pass in multiple context points but this function
		will only calculate the posterior based on the current prior and will not update
		in between each context point.
        """
        if last_TS == None:
            self.TS_probs = [np.nan] * len(self.likelihood_dist)
            return self.TS_probs
        else:
            TS_probs = []
            ld = self.likelihood_dist
            eps = self.TS_eps
            self.history.append(context)
            avg_context = np.average(self.history,weights = [self.k**i for i in range(len(self.history))][::-1])
            likelihood = np.array([dis.pdf(avg_context) for dis in ld])
            likelihood = likelihood/np.sum(likelihood,0)
            TS_probs = likelihood
            TS_probs*= self.bias
            TS_probs = TS_probs/np.sum(TS_probs,0)
            perseverance = np.array([0,0])
            perseverance[last_TS] = 1
            TS_probs = (1-self.perseverance)*TS_probs + self.perseverance*perseverance  # mixed model of TS posteriors and perseverence 
            self.TS_probs = TS_probs
            TS_probs = (1-eps)*TS_probs+eps/2  # mixed model of TS posteriors and random guessing
            return TS_probs
           
    def choose(self, mode = 'e-greedy', eps = None, inv_temp = 1):
        if np.isnan(self.posterior[0]):
            return np.nan
        if eps == None:
            eps = self.TS_eps
        if mode == "e-greedy":
            TS_probs = (1-eps)*self.TS_probs+eps/2
            return np.random.choice(range(len(TS_probs)), p = TS_probs)
        elif mode == 'prob_match':
            return np.random.choice(range(len(self.TS_probs)), p = self.TS_probs)
        elif mode == "softmax":
            probs = softmax(self.posterior, inv_temp)
            return np.random.choice(range(len(probs)), p = probs)
        else:
            return np.argmax(self.TS_probs)
            
            
class SwitchModel:
    """
    Prediction model that takes in data, and uses a prior over hypotheses
    and the relevant to calculate posterior hypothesis estimates.
    """
    def __init__(self, r1 = .9, r2 = .9, eps = 0):
        self.trans_probs = np.array([[r1, 1-r1], [1-r2, r2]]).transpose()
        self.eps = eps
        
    def calc_TS_prob(self, last_choice = -1):
        """
        Calculate the probability of each task set given the previous choice.
        This model assumes that task-sets switch by some probability and no
        other information is available to assess task-set identity. It also 
        assumes that the last choice indicates complete confidence in the previous
        TS
        """
        eps = self.eps
        if last_choice == -1:
            return [.5,.5]
        else:
            return (1-eps)*self.trans_probs[:,last_choice]+eps/2
    
       

            

class DataGenerator:
    """
    creates generator for taskset data based on task-set distributions and
    transition probabilities (defined by recursive_p)
    """
    def __init__(self,ts_dis, recursive_p):
        self.ts_dis = ts_dis
        self.rp = recursive_p
        self.curr_ts = r.choice([0,1])
        self.ts_reps = 1
        self.stim_ids = [(0,2),(0,3),(1,2),(1,3)]
        self.trial_count = 0
        
        
    def gen_data(self):
        rp = self.rp
        ts = self.curr_ts
        dis = self.ts_dis[ts]
        trans_probs = np.array([[rp, 1-rp], [1-rp, rp]])
        bin_boundaries = np.linspace(-1,1,11)
        binned = -1.1 + np.digitize([dis.rvs()],bin_boundaries)*.2
        context_sample = max(-1, min(1, binned[0]))
        trial = {
                'trial_count': self.trial_count,
                'ts': ts,
                'c_dis': {'mean': dis.mean(), 'sd': dis.std()},
                'context': context_sample,
                'stim': r.choice(self.stim_ids)
            }
        if r.random() > trans_probs[ts,ts] or self.ts_reps >= 25:
            self.curr_ts = 1-ts
            self.ts_reps = 1
        else:
            self.ts_reps += 1
        self.trial_count += 1
        return trial
        
