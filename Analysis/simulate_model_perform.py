# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:04:57 2015

@author: Ian
"""
import random as r
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import pylab

#Helper Functions
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
    

#simulation parameters
stim_ids = [(0,2),(0,3),(1,2),(1,3)]
exp_len = 10000
recursive_p = .9
trans_probs = np.array([[recursive_p, 1-recursive_p], [1-recursive_p, recursive_p]])
ts_dis = [norm(.3, .37), norm(-.3,.37)]

#Set up trials
trialList = []    
trial_count = 1
curr_onset = 1 #initial onset
stims = r.sample(stim_ids*int(exp_len * .25),exp_len)
      
trial_states = [1] #start off the function
while abs(np.mean(trial_states)-.5) > .1:
    curr_state = r.choice([0,1])
    trial_states = []
    state_reps = 0
    for trial in range(exp_len):
        trial_states.append(curr_state)
        if r.random() > trans_probs[curr_state,curr_state] or state_reps > 25:
            curr_state = 1-curr_state
            state_reps = 0
        else:
            state_reps += 1
                
bin_boundaries = np.linspace(-1,1,11)

for trial in range(exp_len):
    ts = trial_states[trial]
    dis = ts_dis[ts]
    binned = -1.1 + np.digitize([dis.rvs()],bin_boundaries)*.2
    context_sample = max(-1, min(1, binned[0]))
    trialList += [{
                'trial_count': trial_count,
                'state': trial_states[trial],
                'ts': ts,
                'c_dis': {'mean': dis.mean(), 'sd': dis.std()},
                'context': context_sample,
                'stim': stims[trial]
            }]
    trial_count += 1
trialList = pd.DataFrame(trialList)

#Modeling

prior_ignore = [.5,.5] #base rate fallacy
prior_single = [.5,.5] #changes to reflect state transitions assuming state = argmax(posterior)
prior_optimal = [.5,.5] 

posterior_ignore, posterior_single, posterior_optimal = [],[],[]

for context in trialList.context:
    posterior_ignore.append(calc_posterior(context, prior_ignore,ts_dis))
    posterior_single.append(calc_posterior(context, prior_single,ts_dis))
    posterior_optimal.append(calc_posterior(context, prior_optimal,ts_dis))
    
    prior_single = trans_probs[np.argmax(posterior_single[-1]),:]
    prior_optimal = np.dot(trans_probs,posterior_optimal[-1])
    
m_data = trialList.copy()
m_data['ignore'] = [val[0] for val in posterior_ignore]
m_data['single'] = [val[0] for val in posterior_single]
m_data['optimal'] = [val[0] for val in posterior_optimal]
for model in ['ignore','single','optimal']:
    m_data[model + '_choice'] = [round(val) for val in m_data[model]]
for model in ['ignore','single','optimal']:
    m_data[model + '_noisy'] = [max(min(val+(r.random()*.4-.2),1),0) for val in m_data[model]]
for model in ['ignore','single','optimal']:
    m_data[model + '_noisy_choice'] = [round(val) for val in m_data[model + '_noisy']]

m_data.to_csv('../Data/model_simulation.csv')




#Analysis
ignore_cols = ['c_dis','state','stim','trial_count']
sub = m_data.drop(ignore_cols,1)
pd.scatter_matrix(sub)
pd.scatter_matrix(sub.query('context < .35 and context > -.35'))

sub = m_data
plotting_dict = {'optimal': ['optimal', 'b','optimal'],
                 'single': ['single', 'c','TS(t-1)'],
                  'ignore': ['ignore', 'r','base rate neglect']}
                     
plt.hold(True)
models = []
displacement = 0
#plot model certainty and task-set choices
for arg in plotting_dict.values():
    if arg[2] not in []:
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