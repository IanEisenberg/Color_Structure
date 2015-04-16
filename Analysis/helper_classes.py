# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:09:29 2015

@author: Ian
"""

import random as r
import numpy as np
from scipy.stats import norm

class PredModel:
    """
    Prediction model that takes in data, and uses a prior over hypotheses
    and the relevant to calculate posterior hypothesis estimates.
    """
    def __init__(self, likelihood_dist, prior, recursive_prob = .9,
                 data_noise = 0, mean_noise = 0, std_noise = 0, rp_noise = 0,
                 mode = "optimal"):
        self.prior = prior
        self.likelihood_dist = likelihood_dist
        self.recursive_prob = recursive_prob
        self.dn = data_noise
        self.mn = mean_noise
        self.sn = std_noise
        self.rpn = rp_noise
        self.mode = mode
        self.posterior = prior
        
    def calc_posterior(self, data, noise_val = None, noise = 'gaussian'):
        """
        Calculate the posterior probability of different distribution hypotheses
        given the data. You can specify a few different kinds of noise:
        data_noise = noisy perceptual encoding
        mean_noise = noisy distribution of mean estimates
        std_noise = noisy distribution of std estimates
        The value specified will be a scaling parameter. It should always
        be less than one. You can also speciy
        the noise distribution (default = uniform, alternative = gaussian)
        """
        ld = self.likelihood_dist
        rp = self.recursive_prob
        prior = self.prior
        mode = self.mode
        if not noise_val:
            dn = self.dn
            mn = self.mn
            sn = self.sn
            rpn = self.rpn
        else:
            dn,mn,sn, rpn = np.array([noise_val]*4)*[bool(val) for val in [self.dn,self.mn,self.sn,self.rpn]]
        #noise specification
        #dn changes the perceived data
        #mn and sn change the estimate of the generating distributions, and
        #therefore create noisy likelihood estimates
        if noise.lower() == 'uniform':
            data = min(max(data + dn*r.random()*2-1,-1),1)
            ld = [norm(min(max(dis.mean()+r.random()*mn,-1),1), (dis.std()+r.random()*sn))
                                for dis in ld]
            rp = min(max(rp+r.random()*rpn,0),1)
        elif noise.lower() == 'gaussian':
            data= min(max(data + dn*norm().rvs(),-1),1)
            ld = [norm(min(max(dis.mean()+norm().rvs()*mn,-1),1), abs(dis.std()+norm().rvs()*sn)) 
                                for dis in ld]
            rp = min(max(rp+norm.rvs()*rpn,0),1)
                                    
        trans_probs = np.array([[rp, 1-rp], [1-rp, rp]])    
                     
        n = len(prior)
        likelihood = [dis.pdf(data) for dis in ld]
        numer = np.array([likelihood[i] * prior[i] for i in range(n)])
        try:
            dinom = [np.sum(zip(*numer)[i]) for i in range(len(numer[0]))]
        except TypeError:
            dinom = np.sum(numer)
        posterior = numer/dinom
        
        if mode == "single":
            self.prior = trans_probs[np.argmax(posterior),:]
        elif mode == 'optimal':
            self.prior = np.dot(trans_probs,posterior)
        self.posterior = posterior
        return posterior
        
    def set_mode(self, mode):
        self.mode = mode
        
    def choose(self, mode = None):
        if mode == "noisy":
            return np.random.choice(range(len(self.spoterior)), self.posterior)
        else:
            return np.argmax(self.posterior)
            
            
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
        
