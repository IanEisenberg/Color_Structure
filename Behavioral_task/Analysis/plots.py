# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:08:21 2015

A recording of created, but unneeded plots.

@author: Ian
"""
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pylab
from Load_Data import load_data
import glob
import re

#*********************************************
# Set up helper functions
#*********************************************
def calc_posterior(data,prior,likelihood_dist):
    n = len(prior)
    likelihood = [dis.pdf(data) for dis in likelihood_dist]
    numer = np.array([likelihood[i] * prior[i] for i in range(n)])
    try:
        dinom = [np.sum(list(zip(*numer))[i]) for i in range(len(numer[0]))]
    except TypeError:
        dinom = np.sum(numer)
    posterior = numer/dinom
    return posterior
    
#*********************************************
# Load data
#*********************************************  
train_files = glob.glob('../RawData/*Context_20*yaml')
test_files = glob.glob('../RawData/*Context_noFB*yaml')


data_file = test_files[0]
name = data_file[11:-5]
subj = re.match(r'(\w*)_Prob*', name).group(1)
taskinfo, df, dfa = load_data(data_file, name, mode = 'test')

sub = dfa

recursive_p = taskinfo['recursive_p']
states = taskinfo['states']
state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
trans_probs = np.array([[recursive_p, 1-recursive_p], [1-recursive_p,recursive_p]])
ts_order = [states[0]['ts'],states[1]['ts']]
ts_dis = [norm(states[ts_order[0]]['c_mean'], states[ts_order[0]]['c_sd']),
          norm(states[ts_order[1]]['c_mean'], states[ts_order[1]]['c_sd'])]
#*********************************************
# Plots
#*********************************************


#Plot how optimal inference changes based on context value and priors
x = np.linspace(-1,1,100)
y_biasUp = calc_posterior(x,[.9,.1],ts_dis)
y_even = calc_posterior(x,[.5,.5],ts_dis)
y_biasDown = calc_posterior(x,[.1,.9],ts_dis)
plt.hold(True)
plt.plot(x,y_biasUp[0],lw = 3, label = "prior P(TS1) = .9")
plt.plot(x,y_even[0], lw = 3, label = "prior P(TS1) = .5")
plt.plot(x,y_biasDown[0], lw = 3, label = "prior P(TS1) = .1")
plt.axhline(.5,color = 'y', ls = 'dashed', lw = 2)
plt.xlabel('Stimulus Vertical Position')
plt.ylabel('Posterior P(TS1)')
pylab.legend(loc='upper left')

#sort the context values by state
sub_sorted = sub.sort('ts')
fig = plt.figure()
plt.hold(True)
plt.plot([i*2-1 for i in sub_sorted.ts], 'ro')
plt.plot(sub_sorted.context,'k', lw = 2)
plt.ylabel('Vertical Height')
plt.xlabel('sorted trials')