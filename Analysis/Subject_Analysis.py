# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:12:43 2015

@author: Ian
"""


import yaml
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from Load_Data import load_data
from ggplot import *
import glob

train_files = glob.glob('../Data/*Struct_20*')
test_files = glob.glob('../Data/*Struct_noFB*')

name = train_files[0]
data_file = '../Data/' + name
taskinfo, df, dfa = load_data(data_file, name)

#*********************************************
# Set up plotting defaults
#*********************************************

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18,
        }
        
axes = {'titleweight' : 'bold'
        }
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('figure', figsize = (8,8))

#*********************************************
# Set up helper functions
#*********************************************
def window_function(func, array, window):
    slices = len(array)/window
    windowed_scores = [float(func(array[i*window:(i+1)*window])) for i in range(slices)]
    if slices*window < len(array):
        windowed_scores.append(float(func(array[(slices)*window:])))
    return np.array(windowed_scores)

def bar(x, y, title):
    plot = plt.bar(x,y)
    plt.title(str(title))
    return plot

def calc_posterior(data,prior,likelihood_dist):
    n = len(prior)
    likelihood = [dis.pdf(data) for dis in likelihood_dist]
    numer = np.array([likelihood[i] * prior[i] for i in range(n)])
    dinom = np.sum(numer)
    posterior = numer/dinom
    return posterior
    

#*********************************************
# Preliminary Analysis
#*********************************************

state_dis = [norm(state['c_mean'], state['c_sd']) for state in taskinfo['states'].values()]


#Basic things - look at distribution of RT, etc.
plt.plot(dfa.rt*1000,'ro')
plt.title('RT over experiment', size = 24)
plt.xlabel('trial')
plt.ylabel('RT in ms')


#*********************************************
# Optimal task-set inference 
#*********************************************
state_dis = [norm(state['c_mean'], state['c_sd']) for state in taskinfo['states'].values()]
transitions = np.array([[recursive_p, 1-recursive_p], [1-recursive_p,recursive_p]])

prior_ignore = [.5,.5] #base rate fallacy
prior_single = [.5,.5] #changes to reflect state transitions assuming state = argmax(posterior)
prior_optimal = [.5,.5] 

posterior_ignore, posterior_single, posterior_optimal = [],[],[]

for context in dfa.context:
    posterior_ignore.append(calc_posterior(context[0], prior_ignore,state_dis))
    posterior_single.append(calc_posterior(context[0], prior_single,state_dis))
    posterior_optimal.append(calc_posterior(context[0], prior_optimal,state_dis))
    
    prior_single = transitions[np.argmax(posterior_single[-1]),:]
    prior_optimal = np.dot(transitions,posterior_optimal[-1])

dfa['posterior_ignore'] = posterior_ignore
dfa['posterior_single'] = posterior_single
dfa['posterior_optimal'] = posterior_optimal
       
plt.hold(True)
plt.plot([i*2-1 for i in dfa.state], 'ro')
plt.plot([i[0] for i in dfa.context])

dfa_sorted = dfa.sort('state')
plt.hold(True)
plt.plot([i*2-1 for i in dfa_sorted.state], 'ro')
plt.plot([i[0] for i in dfa_sorted.context])

tmp = dfa[0:len(dfa)/2]
plt.hold(True)
plt.plot(tmp.state, 'ro')
plt.plot([i[1] for i in tmp.posterior_ignore])
plt.plot([i[1] for i in tmp.posterior_single])
plt.plot([i[1] for i in tmp.posterior_optimal])











