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

subj_i = 5
train_file = train_files[subj_i]
test_file = test_files[subj_i]

test_name = test_file[11:-5]
train_name = train_file[11:-5]
subj_name = re.match(r'(\w*)_Prob*', test_name).group(1)

try:
    train_dict = pickle.load(open('../Data/' + train_name + '.p','rb'))
except FileNotFoundError:
    taskinfo, df, dfa = load_data(train_file, train_name, mode = 'train')
    train_dict = {'taskinfo': taskinfo, 'dfa': dfa}
    pickle.dump(train_dict, open('../Data/' + train_name + '.p','wb'))
    
try:
    test_dict = pickle.load(open('../Data/' + test_name + '.p','rb'))
    taskinfo, dfa = [test_dict.get(k) for k in ['taskinfo','dfa']]
except FileNotFoundError:
    taskinfo, df, dfa = load_data(test_file, test_name, mode = 'test')
    test_dict = {'taskinfo': taskinfo, 'dfa': dfa}
    pickle.dump(test_dict, open('../Data/' + test_name + '.p','wb'))


#*********************************************
# Preliminary Setup
#*********************************************
recursive_p = taskinfo['recursive_p']
states = taskinfo['states']
state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
trans_probs = np.array([[recursive_p, 1-recursive_p], [1-recursive_p,recursive_p]])
dfa['abs_context'] = abs(dfa.context)    
train_dfa = train_dict['dfa']
train_dfa['abs_context'] = abs(train_dfa.context)
behav_sum = odict()


#*********************************************
# Generic Experimental Settings
#*********************************************
behav_sum['train_len'] = len(train_dfa)
behav_sum['test_len'] = len(dfa)

#*********************************************
# Performance
#*********************************************    
behav_sum['train_ts1_acc'], behav_sum['train_ts2_acc'] = list(train_dfa.groupby('ts').correct.mean())
behav_sum['test_ts1_acc'], behav_sum['test_ts2_acc'] = list(dfa.groupby('ts').correct.mean())

sub = train_dfa
logit = sm.Logit(sub['correct'], sub[['trial_count','rt','abs_context']])
result = logit.fit()
print(result.summary())
#*********************************************
# Switch costs 
#*********************************************
#RT difference when switching to either action of a new task-set
TS_switch_cost = np.mean(dfa.query('subj_switch == True')['rt']) - np.mean(dfa.query('subj_switch == False')['rt'])
#RT difference when switching to the other action within a task-set
switch_resp_cost = np.mean(dfa.query('rep_resp == False and subj_switch != True')['rt']) - np.mean(dfa.query('rep_resp == True')['rt'])
TS_minus_resp_switch_cost = TS_switch_cost - switch_resp_cost
behav_sum['Switch_cost'] = TS_minus_resp_switch_cost

#*********************************************
# linear fit of RT based on absolute context
#*********************************************

result = sm.GLM(dfa.rt,dfa.abs_context).fit()
behav_sum['context->rt'] = result.params[0]

#*********************************************
# Switch training accuracy
#*********************************************
behav_sum['train_switch_acc'] = train_dfa[int(len(train_dfa)/2):].groupby('subj_switch').correct.mean()[1]


#*********************************************
# estimate of subjective transition probabilities
#*********************************************
subj_recursive_p = (1-dfa.subj_switch.mean())
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

bias = scipy.optimize.curve_fit(fitfunc,dfa,np.ones(len(dfa)), p0 = .2)[0][0]

print(np.argmax(model_likelihoods.sum()))


dfa['abs_context'] = abs(dfa.context)
dfa.groupby('abs_context').mean()
dfa_modeled = pd.concat([dfa,model_posteriors],axis = 1)



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





