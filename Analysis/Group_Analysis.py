# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:16:08 2015

@author: Ian
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from Load_Data import load_data
from helper_classes import PredModel, BiasPredModel, EstimatePredModel
from helper_functions import *
import statsmodels.api as sm
import pickle
import glob
import re
import lmfit
from collections import OrderedDict as odict


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
fullInfo = False

if fullInfo:
   train_files = glob.glob('../RawData/*FullInfo_20*yaml')
   test_files = glob.glob('../RawData/*FullInfo_noFB*yaml') 
else:
    train_files = glob.glob('../RawData/*Context_20*yaml')
    test_files = glob.glob('../RawData/*Context_noFB*yaml')
    
count = 0
for train_file, test_file in zip(train_files,test_files):
    count += 1
    if count != 0:
        pass #continue
    
    test_name = test_file[11:-5]
    train_name = train_file[11:-5]
    subj_name = re.match(r'(\w*)_Prob*', test_name).group(1)
    print(subj_name)
    try:
        train_dict = pickle.load(open('../Data/' + train_name + '.p','rb'))
        taskinfo, train_dfa = [train_dict.get(k) for k in ['taskinfo','dfa']]
    
    except FileNotFoundError:
        train_taskinfo, train_dfa = load_data(train_file, train_name, mode = 'train')
        train_dict = {'taskinfo': train_taskinfo, 'dfa': train_dfa}
        pickle.dump(train_dict, open('../Data/' + train_name + '.p','wb'))
        
    try:
        test_dict = pickle.load(open('../Data/' + test_name + '.p','rb'))
        taskinfo, test_dfa = [test_dict.get(k) for k in ['taskinfo','dfa']]
    except FileNotFoundError:
        taskinfo, test_dfa = load_data(test_file, test_name, mode = 'test')
        test_dict = {'taskinfo': taskinfo, 'dfa': test_dfa}
        pickle.dump(test_dict, open('../Data/' + test_name + '.p','wb'))
        

    
    #*********************************************
    # Preliminary Setup
    #*********************************************
    recursive_p = taskinfo['recursive_p']
    states = taskinfo['states']
    state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
    ts_order = [states[0]['ts'],states[1]['ts']]
    ts_dis = [state_dis[i] for i in ts_order]
    
    #What was the mean contextual value for each taskset during this train run?
    train_ts_means = list(train_dfa.groupby('ts').agg(np.mean).context)
    #Same for standard deviation
    train_ts_std = list(train_dfa.groupby('ts').agg(np.std).context)
    train_ts_dis = [norm(m,s) for m,s in zip(train_ts_means,train_ts_std)]
    #And do the same for recursive_p
    train_recursive_p = 1- train_dfa.switch.mean()
    
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
    #deterministic feedback). Basically, after receiving FB, the ideal observer
    #knows exactly what task it is in and should act accordingly.
    
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
    optimal_observer = BiasPredModel(train_ts_dis, [.5,.5], recursive_prob = train_recursive_p)
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
    
    #accuracy is defined in relation to the ideal observer
    behav_sum['train_ts1_acc'], behav_sum['train_ts2_acc'] = list(train_dfa.groupby('ts').conform_observer.mean())
    behav_sum['test_ts1_acc'], behav_sum['test_ts2_acc'] = list(test_dfa.groupby('ts').conform_observer.mean())
    
    #Very course estimate of learning: is there a change in performance over trials?
    #Threshold p < .01, and if so, what direction?
    learn_direct = []
    for sub in [train_dfa, test_dfa]:
        logit = sm.Logit(sub['conform_observer'], sm.add_constant(sub[['trial_count']]))
        result = logit.fit()
        learn_direct.append(int(result.pvalues[1]<.01) * np.sign(result.params[1]))
    behav_sum['learning?'] = learn_direct
    
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
    
    result = sm.GLS(test_dfa.rt,sm.add_constant(test_dfa.abs_context)).fit()
    behav_sum['context->rt'] = result.params[1] * int(result.pvalues[1]<.05)
    
    
    #*********************************************
    # Switch training accuracy
    #*********************************************
    
    behav_sum['train_switch_acc'] = train_dfa.groupby('subj_switch').conform_observer.mean()[1]
    
    #*********************************************
    # Contributors to task-set choice
    #*********************************************
    sub = sm.add_constant(test_dfa[['context','subj_ts','rt']])
    sub['last_ts'] = sub.subj_ts.shift(1)
    predictors = sub.drop(['subj_ts'],axis = 1)
    logit = sm.Logit(sub['subj_ts'],predictors, missing = 'drop')
    result = logit.fit()
    print(result.summary())
    
    #*********************************************
    # Model fitting
    #*********************************************
    
    #*************************************
    #Model Functions
    #*************************************
    
    def bias_fitfunc(dfa, rp, tsb):
        model = BiasPredModel(train_ts_dis, init_prior, ts_bias = tsb, recursive_prob = rp)
        model_likelihoods = []
        for i,trial in dfa.iterrows():
            c = trial.context
            trial_choice = trial.subj_ts
            conf = model.calc_posterior(c)
            model_likelihoods.append(conf[trial_choice])
        return model_likelihoods
        
    def bias_errfunc(dfa,rp,tsb):
        return (bias_fitfunc(dfa,rp,tsb) - np.ones(len(dfa)))
    
    def bias_fit_temp(dfa, tsb, rp, t):
        model = BiasPredModel(train_ts_dis, init_prior, ts_bias = tsb, recursive_prob = rp, temp = t)
        model_choices = []
        for i,trial in dfa.iterrows():
            c = trial.context
            model.calc_posterior(c)
            model_choices.append(model.choose())
        dfa['model_choices'] = model_choices
        return dfa.groupby('context').model_choices.mean()
        
    def estimate_fitfunc(dfa, m,s,rp):
        model = EstimatePredModel(init_prior, mean = m, std = s, recursive_prob = rp)
        model_likelihoods = []
        for i,trial in dfa.iterrows():
            c = trial.context
            trial_choice = trial.subj_ts
            conf = model.calc_posterior(c)
            model_likelihoods.append(conf[trial_choice])
        return model_likelihoods    
    
    def estimate_errfunc(dfa,m,s,rp):
        return (estimate_fitfunc(dfa,m,s,rp) - np.ones(len(dfa)))    
        
    def estimate_fit_temp(dfa, m,s,rp, t):
        model = EstimatePredModel(init_prior,mean = m, std = s, recursive_prob = rp, temp = t)
        model_choices = []
        for i,trial in dfa.iterrows():
            c = trial.context
            model.calc_posterior(c)
            model_choices.append(model.choose())
        dfa['model_choices'] = model_choices
        return dfa.groupby('context').model_choices.mean()
        
        
        
    init_prior = [.5,.5]
    
    #Fit bias model
    model = lmfit.Model(bias_fitfunc, independent_vars = ['dfa'])    
    params = lmfit.Parameters()
    params.add('tsb', value = 1, min = 0)
    params.add('rp', value = train_recursive_p)
    modelfit = model.fit(np.ones(len(test_dfa)), params, dfa = test_dfa)
    tsb, brp = [modelfit.values.get(k) for k in ['tsb', 'rp']]
    behav_sum['bias_fit_params'] = modelfit.values
    behav_sum['bias_fit_error'] = np.sum(np.square(bias_errfunc(test_dfa,brp,tsb)))
    #Fit softmax temp
    model = lmfit.Model(bias_fit_temp, independent_vars = ['dfa','tsb','rp'])    
    params = lmfit.Parameters()
    params.add('t', min = .01)
    modelfit = model.fit(test_dfa.groupby('context').subj_ts.mean(), params, dfa = test_dfa, tsb = tsb, rp = brp)
    btemp = modelfit.values['t'] 
        
    #Fit Estimate model
    model = lmfit.Model(estimate_fitfunc, independent_vars = ['dfa'])    
    params = lmfit.Parameters()
    params.add('m', value = train_ts_dis[0].mean(), min = -1, max = 1)
    params.add('s', value = train_ts_dis[0].std())
    params.add('rp', value = train_recursive_p)
    modelfit = model.fit(np.ones(len(test_dfa)), params, dfa = test_dfa)
    m, s, erp = [modelfit.values.get(k) for k in ['m','s','rp']]
    behav_sum['estimate_fit_params'] = modelfit.values
    behav_sum['estimate_fit_error'] = np.sum(np.square(estimate_errfunc(test_dfa,m,s,erp)))
    #Fit softmax temp
    model = lmfit.Model(estimate_fit_temp, independent_vars = ['dfa','m','s','rp'])    
    params = lmfit.Parameters()
    params.add('t', min = .01)
    modelfit = model.fit(test_dfa.groupby('context').subj_ts.mean(), params, dfa = test_dfa, m=m, s=s, rp=erp)
    etemp = modelfit.values['t']
    
    #record temp settings
    behav_sum['softmax_temps'] = {'bias_temp':btemp, 'estimate_temp':etemp}
     
     
    model_choice = ['ignore','single','optimal']
    models = [ \
        PredModel(train_ts_dis, init_prior, mode = "ignore", recursive_prob = train_recursive_p),\
        PredModel(train_ts_dis, init_prior, mode = "single", recursive_prob = train_recursive_p),\
        PredModel(train_ts_dis, init_prior, mode = "optimal", recursive_prob = train_recursive_p)]
        
    model_posteriors = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
    model_choices = pd.DataFrame(columns = ['ignore','single','optimal'], dtype = 'float64')
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
            trial_model_likelihoods += [conf[trial_choice]]
        #add on 'straw model' predictions.
        trial_model_likelihoods += [.5,[.9,.1][trial_choice], [.1,.9][trial_choice]] 
       
       #record trial estimates
        model_likelihoods.loc[i] = np.log(trial_model_likelihoods)
        model_posteriors.loc[i] = model_posterior
        model_choices.loc[i] = model_choice
        
    behav_sum['best_model'] = np.argmax(model_likelihoods.sum())


    #*********************************************
    # Add to group dictionary
    #*********************************************
    group_behavior[subj_name] = behav_sum
    
    
    
group_df = pd.DataFrame(group_behavior).transpose()   
group_df = group_df[list(behav_sum.keys())]
    
        