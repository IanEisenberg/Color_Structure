# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:16:08 2015

@author: Ian
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from Load_Data import load_data
from helper_classes import PredModel
from helper_functions import *
from ggplot import *
import statsmodels.api as sm
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
group_behavior = {}
train_files = glob.glob('../RawData/*Context_20*yaml')
test_files = glob.glob('../RawData/*Context_noFB*yaml')

count = 0
for train_file, test_file in zip(train_files,test_files):
    count += 1
    if count != 999:
        pass #continue
    
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
        
    
    recursive_p = taskinfo['recursive_p']
    states = taskinfo['states']
    state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
    trans_probs = np.array([[recursive_p, 1-recursive_p], [1-recursive_p,recursive_p]])
    dfa['abs_context'] = abs(dfa.context)    
    train_dfa = train_dict['dfa']
    behav_sum = {}
    
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
    
    result = sm.GLM(dfa.rt,dfa.context).fit()
    behav_sum['context->rt'] = result.params[0]
    
    #*********************************************
    # Switch training accuracy
    #*********************************************
    behav_sum['switch_acc'] = train_dfa[int(len(train_dfa)/2):].groupby('subj_switch').correct.mean()[1]

    
    #*********************************************
    # estimate of subjective transition probabilities
    #*********************************************
    subj_recursive_p = (1-dfa.subj_switch.mean())
    train_recursive_p = (1-train_dfa.switch.mean())
    behav_sum['subj_recursive_p'] = subj_recursive_p/train_recursive_p
    behav_sum['train_statistics'] = {'recursive_p':train_recursive_p}
    
    #*********************************************
    # Test Accuracy as proportion of optimal model
    #*********************************************
    
    #When subjects performed consistently with a particular TS, what was the mean context value?
    experienced_ts_means = list(train_dfa.groupby('subj_ts').agg(np.mean).context)
    #Same for standard deviation
    experienced_ts_std = list(train_dfa.groupby('subj_ts').agg(np.std).context)
    behav_sum['train_statistics']['ts_mean_ts'] = list(zip(experienced_ts_means,experienced_ts_std))    
    
    ts_dis = [norm(mean,std) for mean,std in zip(experienced_ts_means,experienced_ts_std)]

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
    behav_sum['best_model'] = np.argmax(model_likelihoods.sum())
    
    #Calculated 'task-set' accuracy. Coded as correct as long as the response conformed
    #to the appropriate task-set (thus there could be stim level errors)
    ts_acc = np.mean(np.equal(dfa.ts,dfa.subj_ts))
    optimal_ts_acc = np.mean(np.equal(model_choices.optimal,dfa.ts))
    behav_sum['ts_acc'] = ts_acc/optimal_ts_acc

    
    #*********************************************
    # Add to group dictionary
    #*********************************************
    group_behavior[subj_name] = behav_sum
    
    
    
group_df = pd.DataFrame(group_behavior).transpose()   
    
    