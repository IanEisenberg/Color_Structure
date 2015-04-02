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

train_files = glob.glob('../Data/*Struct_20*yaml')
test_files = glob.glob('../Data/*Struct_noFB*yaml')

data_file = train_files[2]
name = data_file[8:-5]
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
plt.rc('figure', figsize = (12,12))

#*********************************************
# Set up helper functions
#*********************************************
def track_runs(iterable):
    """
    Return the item with the most consecutive repetitions in `iterable`.
    If there are multiple such items, return the first one.
    If `iterable` is empty, return `None`.
    """
    track_repeats=[]
    current_element = None
    current_repeats = 0
    element_i = 0
    for element in iterable:
        if current_element == element:
            current_repeats += 1
        else:
            track_repeats.append((current_repeats,current_element, element_i-current_repeats))
            current_element = element
            current_repeats = 1
        element_i += 1
    return track_repeats
    
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
recursive_p = taskinfo['recursive_p']
states = taskinfo['states']
state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]


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
    posterior_ignore.append(calc_posterior(context, prior_ignore,state_dis))
    posterior_single.append(calc_posterior(context, prior_single,state_dis))
    posterior_optimal.append(calc_posterior(context, prior_optimal,state_dis))
    
    prior_single = transitions[np.argmax(posterior_single[-1]),:]
    prior_optimal = np.dot(transitions,posterior_optimal[-1])

dfa['posterior_ignore'] = posterior_ignore
dfa['posterior_single'] = posterior_single
dfa['posterior_optimal'] = posterior_optimal
   
   
 #smooth the posterior estimates using an exponentially-weighted moving average
span_val = 3
dfa['smoothed_ignore']=pd.stats.moments.ewma(pd.Series([i[1] for i in dfa.posterior_ignore]), span = span_val)
dfa['smoothed_single']=pd.stats.moments.ewma(pd.Series([i[1] for i in dfa.posterior_single]), span = span_val)
dfa['smoothed_optimal']=pd.stats.moments.ewma(pd.Series([i[1] for i in dfa.posterior_optimal]), span = span_val)
#smooth context by same value 
dfa['smoothed_context']=pd.stats.moments.ewma(pd.Series(dfa.context), span = span_val)


#*********************************************
# Plotting
#*********************************************
#plot context values and show the current state    
plt.hold(True)
plt.plot([i*2-1 for i in dfa.ts], 'ro')
plt.plot(dfa.context)

#sort the context values by state
dfa_sorted = dfa.sort('ts')
plt.hold(True)
plt.plot([i*2-1 for i in dfa_sorted.ts], 'ro')
plt.plot(dfa_sorted.context)

#plot the optimal posterior estimate against a base-rate ignoring model
plt.hold(True)
plt.plot(tmp.ts, 'ro')
plt.plot(dfa['smoothed_ignore'])
plt.plot(dfa['smoothed_optimal'])

#plot the posterior estimate on top of the actual context
plt.hold(True)
plt.plot([i*2-1 for i in dfa.ts], 'ro')
plt.plot(dfa['smoothed_context'])
plt.plot(dfa['smoothed_ignore'], linewidth = 3)
plt.plot(dfa['smoothed_optimal'], linewidth = 3)

#unsmoothed
plt.hold(True)
plt.plot([i*2-1 for i in dfa.ts], 'ro')
plt.plot(dfa['context'])
plt.plot([i[0] for i in dfa['posterior_ignore']], linewidth = 3)
plt.plot([i[0] for i in dfa['posterior_optimal']], linewidth = 3)







