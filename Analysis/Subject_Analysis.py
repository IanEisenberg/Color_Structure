# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:12:43 2015

@author: Ian
"""


import numpy as np
from scipy.stats import norm
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from Load_Data import load_data
import statsmodels.api as sm
from collections import OrderedDict as odict
from helper_classes import PredModel, BiasPredModel
from helper_functions import *
from ggplot import *
import pickle
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
# Load Data
#*********************************************
    
train_files = glob.glob('../RawData/*Context_20*yaml')
test_files = glob.glob('../RawData/*Context_noFB*yaml')

subj_i = 4
train_file = train_files[subj_i]
test_file = test_files[subj_i]

test_name = test_file[11:-5]
train_name = train_file[11:-5]
subj_name = re.match(r'(\w*)_Prob*', test_name).group(1)

try:
    train_dict = pickle.load(open('../Data/' + train_name + '.p','rb'))
    taskinfo, train_dfa = [train_dict.get(k) for k in ['taskinfo','dfa']]

except FileNotFoundError:
    train_taskinfo, train_dfa = load_data(train_file, train_name, mode = 'train')
    pickle.dump(train_dict, open('../Data/' + train_name + '.p','wb'))
    
try:
    test_dict = pickle.load(open('../Data/' + test_name + '.p','rb'))
    taskinfo, test_dfa = [test_dict.get(k) for k in ['taskinfo','dfa']]
except FileNotFoundError:
    taskinfo, test_dfa = load_data(test_file, test_name, mode = 'test')
    test_dict = {'taskinfo': taskinfo, 'dfa': dfa}
    pickle.dump(test_dict, open('../Data/' + test_name + '.p','wb'))


#*********************************************
# Preliminary Setup
#*********************************************
recursive_p = taskinfo['recursive_p']
states = taskinfo['states']
state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
ts_order = [states[0]['ts'],states[1]['ts']]
ts_dis = [state_dis[i] for i in ts_order]

test_dfa['abs_context'] = abs(test_dfa.context)    
train_dfa['abs_context'] = abs(train_dfa.context)

behav_sum = odict()


#*********************************************
# Set up perfect observer
#*********************************************

#This observer know the exact statistics of the task, always chooses correctly
#given that it chooses the correct task-set, and perfectly learns from feedback.
#This means that it sets the prior probability for each ts to the transition probabilities
#of the correct task-set on each trial (which a subject 'could' do due to the
#deterministic feedback)

for dfa in [train_dfa]:
    observer_prior = [.5,.5]
    observer_choices = []
    for i,trial in dfa.iterrows():
        c = trial.context
        ts = trial.ts
        conf= calc_posterior(c,observer_prior,ts_dis)    
        obs_choice = np.argmax(conf)
        observer_choices.append(obs_choice)
        observer_prior = np.round([.9*(1-ts)+.1*ts,.9*ts+.1*(1-ts)],2)
        
    dfa['observer_choices'] = observer_choices
    dfa['observer_switch'] = abs(dfa.observer_choices.diff())
    dfa['conform_observer'] = np.equal(train_dfa.subj_ts, observer_choices)

#Optimal observer for test        
optimal_observer = BiasPredModel(ts_dis, [.5,.5], bias = 0, recursive_prob = recursive_p)
observer_choices = []
for i,trial in test_dfa.iterrows():
    c = trial.context
    conf = optimal_observer.calc_posterior(c)
    obs_choice = np.argmax(conf)
    observer_choices.append(obs_choice)
test_dfa['observer_choices'] = observer_choices
test_dfa['observer_switch'] = abs(test_dfa.observer_choices.diff())
test_dfa['conform_observer'] = np.equal(test_dfa.subj_ts, observer_choices)

#*********************************************
# Generic Experimental Settings
#*********************************************
behav_sum['train_len'] = len(train_dfa)
behav_sum['test_len'] = len(test_dfa)

#*********************************************
# Performance
#*********************************************    
#accuracy is defined in relation to the observer

behav_sum['train_ts1_acc'], behav_sum['train_ts2_acc'] = list(train_dfa.groupby('ts').conform_observer.mean())
behav_sum['test_ts1_acc'], behav_sum['test_ts2_acc'] = list(test_dfa.groupby('ts').conform_observer.mean())

logit = sm.Logit(sub['correct'], sub[['trial_count']])
result = logit.fit()
print(result.summary())
#*********************************************
# Switch costs 
#*********************************************
#RT difference when switching to either action of a new task-set
TS_switch_cost = np.mean(test_dfa.query('subj_switch == True')['rt']) - np.mean(test_dfa.query('subj_switch == False')['rt'])
#RT difference when switching to the other action within a task-set
switch_resp_cost = np.mean(test_dfa.query('rep_resp == False and subj_switch != True')['rt']) - np.mean(test_dfa.query('rep_resp == True')['rt'])
TS_minus_resp_switch_cost = TS_switch_cost - switch_resp_cost
behav_sum['Switch_cost'] = TS_minus_resp_switch_cost

#*********************************************
# linear fit of RT based on absolute context
#*********************************************

result = sm.GLM(test_dfa.rt,test_dfa.abs_context).fit()
behav_sum['context->rt'] = result.params[0]

#*********************************************
# Switch training accuracy
#*********************************************
behav_sum['train_switch_acc'] = train_dfa[int(len(train_dfa)/2):].groupby('subj_switch').correct.mean()[1]


#*********************************************
# estimate of subjective transition probabilities
#*********************************************
subj_recursive_p = (1-test_dfa.subj_switch.mean())
train_recursive_p = (1-train_dfa.switch.mean())
behav_sum['subj_recursive_p'] = subj_recursive_p
behav_sum['train_statistics'] = {'recursive_p':train_recursive_p}
    
#*********************************************
# Test Accuracy as proportion of optimal model
#*********************************************

#When subjects performed consistently with a particular TS, what was the mean context value?
experienced_ts_means = list(train_dfa.groupby('subj_ts').agg(np.mean).context)
#Same for standard deviation
experienced_ts_std = list(train_dfa.groupby('subj_ts').agg(np.std).context)
behav_sum['train_statistics']['ts_mean_ts'] = list(zip(experienced_ts_means,experienced_ts_std))  

    
#*********************************************
# Optimal task-set inference 
#*********************************************
ts_order = [states[0]['ts'],states[1]['ts']]
ts_dis = [state_dis[i] for i in ts_order]


init_prior = [.5,.5]
model_choice = ['ignore','single','optimal']
models = [ \
    PredModel(ts_dis, init_prior, mode = "ignore", recursive_prob = train_recursive_p),\
    PredModel(ts_dis, init_prior, mode = "single", recursive_prob = train_recursive_p),\
    PredModel(ts_dis, init_prior, mode = "optimal", recursive_prob = train_recursive_p)]
    
model_posteriors = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
model_choices = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
model_prob_matches = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
model_likelihoods = pd.DataFrame(columns = ['ignore','single','optimal','rand','ts0','ts1'], dtype = 'float64')

for i,trial in test_dfa.iterrows():
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

ts_dis = [norm(mean,std) for mean,std in zip(experienced_ts_means,experienced_ts_std)]
def fitfunc(dfa, b):
    model = BiasPredModel(ts_dis, init_prior, bias = b, recursive_prob = train_recursive_p)
    bias_model_likelihoods = []
    for i,trial in dfa.iterrows():
        c = trial.context
        trial_choice = trial.subj_ts
        conf = model.calc_posterior(c)
        bias_model_likelihoods.append(conf[trial_choice])
    return bias_model_likelihoods
    
def errfunc(dfa,b):
    return (fitfunc(dfa,b) - np.ones(len(dfa)))

bias = scipy.optimize.curve_fit(fitfunc,test_dfa,np.ones(len(test_dfa)), p0 = .2)[0][0]

print(np.argmax(model_likelihoods.sum()))


dfa['abs_context'] = abs(test_dfa.context)
test_dfa.groupby('abs_context').mean()
test_dfa_modeled = pd.concat([test_dfa,model_posteriors],axis = 1)



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
        plt.savefig('../Plots/' +  subj_name + '_summary_plot.png', dpi = 300, bbox_inches='tight')





