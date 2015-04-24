# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:12:43 2015

@author: Ian
"""


import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from Load_Data import load_data
from helper_classes import PredModel
from ggplot import *
import glob
import re

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

save = False
plot = False

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

def softmax(probs, temp):
    return np.exp(probs/temp)/sum(np.exp(probs/temp))
    
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
    
def plot_run(sub,plotting_dict, exclude = []):
    #plot the posterior estimates for different models, the TS they currently select
    #and the vertical position of the stimulus
    plt.hold(True)
    models = []
    displacement = 0
    #plot model certainty and task-set choices
    for arg in plotting_dict.values():
        if arg[0] not in exclude:
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
    
#*********************************************
# Load Data
#*********************************************
    
train_files = glob.glob('../RawData/*Context_20*yaml')
test_files = glob.glob('../RawData/*Context_noFB*yaml')


data_file = test_files[1]
name = data_file[11:-5]
subj = re.match(r'(\w*)_Prob*', name).group(1)
taskinfo, df, dfa = load_data(data_file, name, mode = 'test')



#*********************************************
# Preliminary Setup
#*********************************************
recursive_p = taskinfo['recursive_p']
states = taskinfo['states']
state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
trans_probs = np.array([[recursive_p, 1-recursive_p], [1-recursive_p,recursive_p]])

#*********************************************
# Switch costs 
#*********************************************
#RT difference when switching to either action of a new task-set
TS_switch_cost = np.mean(dfa.query('subj_switch == True')['rt']) - np.mean(dfa.query('subj_switch == False')['rt'])
#RT difference when switching to the other action within a task-set
switch_resp_cost = np.mean(dfa.query('rep_resp == False and subj_switch != True')['rt']) - np.mean(dfa.query('rep_resp == True')['rt'])



#*********************************************
# Optimal task-set inference 
#*********************************************
ts_order = [states[0]['ts'],states[1]['ts']]
ts_dis = [norm(states[ts_order[0]]['c_mean'], states[ts_order[0]]['c_sd']),
          norm(states[ts_order[1]]['c_mean'], states[ts_order[1]]['c_sd'])]

init_prior = [.5,.5]
model_choice = ['ignore','single','optimal']
models = [ \
    PredModel(ts_dis, init_prior, mode = "ignore"),\
    PredModel(ts_dis, init_prior, mode = "single"),\
    PredModel(ts_dis, init_prior, mode = "optimal")]
    
model_posteriors = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
model_choices = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
model_prob_matches = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
model_likelihoods = pd.DataFrame(columns = ['ignore','single','optimal','rand','ts0','ts1'], dtype = 'float64')

for i,trial in dfa.iterrows():
    c = trial.context
    trial_choice = trial.subj_ts
    
    model_posterior= []
    model_choice=[]
    model_prob_match = []
    trial_model_likelihoods = []
    for j,model in enumerate(models):
        conf = model.calc_posterior(c)
        model_posterior += [conf[0]]
        model_choice += [model.choose()]
        model_prob_match += [model.choose('softmax')]
        trial_model_likelihoods += [conf[trial_choice]]
    #add on 'straw model' predictions.
    trial_model_likelihoods += [.5,[.9,.1][trial_choice], [.1,.9][trial_choice]] 
   
   #record trial estimates
    model_likelihoods.loc[i] = np.log(trial_model_likelihoods)
    model_posteriors.loc[i] = model_posterior
    model_choices.loc[i] = model_choice
    model_prob_matches.loc[i] = model_prob_match
    

print(np.argmax(model_likelihoods.sum()))


dfa['abs_context'] = abs(dfa.context)
dfa.groupby('abs_context').mean()
dfa_modeled = pd.concat([dfa,model_posteriors],axis = 1)


if save == True:
    dfa.to_csv('../Data/' + name + '_modeled.csv')


#*********************************************
# Plotting
#*********************************************


if plot == True:
    
    #look at RT
    plt.subplot(4,1,1)
    plt.plot(dfa.rt*1000,'ro')
    plt.title('RT over experiment', size = 24)
    plt.xlabel('trial')
    plt.ylabel('RT in ms')
    
    plt.subplot(4,1,2)
    plt.hold(True)
    dfa.query('subj_switch == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'stay')
    dfa.query('subj_switch == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'switch')
    dfa.query('subj_switch == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    dfa.query('subj_switch == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('count')
    pylab.legend(loc='upper right',prop={'size':20})
    
    plt.subplot(4,1,3)
    plt.hold(True)
    dfa.query('subj_switch == 0 and rep_resp == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'repeat response')
    dfa.query('subj_switch == 0 and rep_resp == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'chance response (within task-set)')
    dfa.query('subj_switch == 0 and rep_resp == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    dfa.query('subj_switch == 0 and rep_resp == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('count')
    pylab.legend(loc='upper right',prop={'size':20})
    
    plt.subplot(4,1,4)
    plt.hold(True)
    dfa.query('subj_ts == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'ts1')
    dfa.query('subj_ts == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'ts2')
    dfa.query('subj_ts == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    dfa.query('subj_ts == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('count')
    pylab.legend(loc='upper right',prop={'size':20})
            
    ggplot(dfa, aes(x='context',y='rt', color = 'subj_ts')) + geom_point(alpha=.4) + stat_summary(size = 6)
    ggplot(dfa, aes(x='context',y='correct', color = 'subj_ts')) + stat_summary()
    

    #Plot run
    plotting_dict = {'optimal': ['optimal', 'b','optimal'],
                    'single': ['single', 'c','TS(t-1)'],
                     'ignore': ['ignore', 'r','base rate neglect']}
    sub = dfa_modeled[50:200]
    plot_run(sub,plotting_dict, exclude = ['single'])
    if save == True:
        plt.savefig('../Plots/' +  subj + '_summary_plot.png', dpi = 300, bbox_inches='tight')





