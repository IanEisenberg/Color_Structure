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
import pylab
from Load_Data import load_data
from ggplot import *
import glob

#*********************************************
# Set up plotting defaults
#*********************************************

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20,
        }
        
axes = {'titleweight' : 'bold'
        }
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('figure', figsize = (16,12))

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
    plot = plt.bar(x,y,width = .5)
    plt.title(str(title))
    return plot

def calc_posterior(data,prior,likelihood_dist):
    n = len(prior)
    likelihood = [dis.pdf(data) for dis in likelihood_dist]
    numer = np.array([likelihood[i] * prior[i] for i in range(n)])
    try:
        dinom = [np.sum(zip(*numer)[i]) for i in range(len(numer[0]))]
    except TypeError:
        dinom = np.sum(numer)
    posterior = numer/dinom
    return posterior
    
    
#*********************************************
# Load Data
#*********************************************
    
train_files = glob.glob('../Data/*Struct_20*yaml')
test_files = glob.glob('../Data/*Struct_noFB*yaml')

data_file = test_files[1]
name = data_file[8:-5]
subj = re.match(r'(\w*)_Color*', name).group(1)
taskinfo, df, dfa = load_data(data_file, name, mode = 'test')



#*********************************************
# Preliminary Analysis
#*********************************************
recursive_p = taskinfo['recursive_p']
states = taskinfo['states']
state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
transitions = np.array([[recursive_p, 1-recursive_p], [1-recursive_p,recursive_p]])


#Basic things - look at distribution of RT, etc.
plt.plot(dfa.rt*1000,'ro')
plt.title('RT over experiment', size = 24)
plt.xlabel('trial')
plt.ylabel('RT in ms')


#*********************************************
# Optimal task-set inference 
#*********************************************
ts_order = [states[0]['ts'],states[1]['ts']]
ts_dis = [norm(states[ts_order[0]]['c_mean'], states[ts_order[0]]['c_sd']),
          norm(states[ts_order[1]]['c_mean'], states[ts_order[1]]['c_sd'])]



prior_ignore = [.5,.5] #base rate fallacy
prior_single = [.5,.5] #changes to reflect state transitions assuming state = argmax(posterior)
prior_optimal = [.5,.5] 

posterior_ignore, posterior_single, posterior_optimal = [],[],[]

for context in dfa.context:
    posterior_ignore.append(calc_posterior(context, prior_ignore,ts_dis))
    posterior_single.append(calc_posterior(context, prior_single,ts_dis))
    posterior_optimal.append(calc_posterior(context, prior_optimal,ts_dis))
    
    prior_single = transitions[np.argmax(posterior_single[-1]),:]
    prior_optimal = np.dot(transitions,posterior_optimal[-1])

#The posterior likelihood of ts 0
dfa['ts0_posterior_ignore'] = [val[0] for val in posterior_ignore]
dfa['ts0_posterior_single'] = [val[0] for val in posterior_single]
dfa['ts0_posterior_optimal'] = [val[0] for val in posterior_optimal]
   
   
 #smooth the posterior estimates using an exponentially-weighted moving average
span_val = 3
dfa['smoothed_ts0_ignore']=pd.stats.moments.ewma(dfa.ts0_posterior_ignore, span = span_val)
dfa['smoothed_ts0_single']=pd.stats.moments.ewma(dfa.ts0_posterior_single, span = span_val)
dfa['smoothed_ts0_optimal']=pd.stats.moments.ewma(dfa.ts0_posterior_optimal, span = span_val)
#smooth context by same value 
dfa['smoothed_context']=pd.stats.moments.ewma(pd.Series(dfa.context), span = span_val)

dfa.to_csv('../Data/' + name + '_modeled.csv')


#*********************************************
# Plotting
#*********************************************

plotting_dict = {'optimal': ['ts0_posterior_optimal', 'b','optimal'],
                'single': ['ts0_posterior_single', 'c','TS(t-1)'],
                 'ignore': ['ts0_posterior_ignore', 'r','base rate neglect']}
sub = dfa[150:250]
#plot context values and show the current state    
plt.hold(True)
plt.plot([i*2-1 for i in sub.ts], 'ro')
plt.plot(sub.context, 'k', lw = 2)
plt.ylabel('Vertical Height')
plt.xlabel('trial')
plt.savefig('context_over_trials.png', dpi = 300)

#sort the context values by state
sub_sorted = sub.sort('ts')
fig = plt.figure()
plt.hold(True)
plt.plot([i*2-1 for i in sub_sorted.ts], 'ro')
plt.plot(sub_sorted.context,'k', lw = 2)
plt.ylabel('Vertical Height')
plt.xlabel('sorted trials')
plt.savefig('sorted_context.png', dpi = 300)

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
plt.savefig('../Plots/effect_of_prior.png', dpi = 300)


#plot the posterior estimates for different models, the TS they currently select
#and the vertical position of the stimulus
plt.hold(True)
models = []
displacement = 0
#plot model certainty and task-set choices
for arg in plotting_dict.values():
    if arg[2] not in ['']:
        plt.plot(sub.trial_count,sub[arg[0]]*2,arg[1], label = arg[2], lw = 2)
        plt.plot(sub.trial_count, [int(val>.5)+3+displacement for val in sub[arg[0]]],arg[1]+'o')
        displacement+=.15
        models.append(arg[0])
plt.axhline(1, color = 'y', ls = 'dashed', lw = 2)
plt.axhline(2.5, color = 'k', ls = 'dashed', lw = 3)

#plot subject choices (con_shape = conforming to TS1)
#plot current TS, flipping bit to plot correctly
plt.plot(sub.trial_count,(1-sub.ts)-2, 'go', label = 'operating TS')
plt.plot(sub.trial_count, sub.context/2-1.5,'k', lw = 2, label = 'stimulus height')
plt.plot(sub.trial_count, sub.con_shape+2.85, 'yo', label = 'subject choice')
plt.yticks([-2, -1.5, -1, 0, 1, 2, 3.1, 4.1], [ -1, 0 , 1,'0%', '50%',  '100%', 'TS2 Choice', 'TS1 Choice'])

plt.xlim([min(sub.index)-.5,max(sub.index)])
plt.ylim(-2.5,5)
#subdivide graph
plt.axhline(-.5, color = 'k', ls = 'dashed', lw = 3)
plt.axhline(-1.5, color = 'y', ls = 'dashed', lw = 2)
#axes labels
plt.xlabel('trial number')
plt.ylabel('Predicted P(TS1)')
ax = plt.gca()
ax.yaxis.set_label_coords(-.1, .45)
pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08),
          ncol=3, fancybox=True, shadow=True)

plt.savefig('../Plots/' +  subj + '_summary_plot.png', dpi = 300, bbox_inches='tight')





y = np.linspace(-1,1,100)
x1 = -ts_dis[0].pdf(y)
#x2 = -ts_dis[1].pdf(y)
f = plt.figure(figsize = (6,12))
ax = f.add_subplot(111)
plt.plot(x1,y, lw = 3)
plt.plot(x2,y, lw = 3)
ax.yaxis.tick_right()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_axis_bgcolor((.9,.9,.9))
plt.savefig('../Plots/position_distributions.png', dpi = 300, transparent = True)



