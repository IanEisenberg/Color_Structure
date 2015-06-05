
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
import seaborn as sns
from ggplot import *
from collections import OrderedDict as odict


#*********************************************
# Load Data
#*********************************************

group_behavior = {}
gtrain_df = pd.DataFrame()
gtest_df = pd.DataFrame()
gtaskinfo = []

bias_fit_dict = pickle.load(open('Analysis_Output/bias_parameter_fits.p','rb'))
nobias_fit_dict = pickle.load(open('Analysis_Output/nobias_parameter_fits.p','rb'))

train_files = glob.glob('../Data/*Context_20*.p') 
test_files = glob.glob('../Data/*Context_test*.p') 
    
for train_file, test_file in zip(train_files,test_files):
    subj_name = re.match(r'.*/Data\\(\w*)_Prob*', test_file).group(1)
    print(subj_name)
    
    try:
        train_dict = pickle.load(open(train_file,'rb'))
        taskinfo, train_dfa = [train_dict.get(k) for k in ['taskinfo','dfa']]
    except FileNotFoundError:
        print('Train file not found')
    try:
        test_dict = pickle.load(open(test_file,'rb'))
        taskinfo, test_dfa = [test_dict.get(k) for k in ['taskinfo','dfa']]
    except FileNotFoundError:
        print('Test file not found')

#*********************************************
# Preliminary Setup
#*********************************************

    
    recursive_p = taskinfo['recursive_p']
    states = taskinfo['states']
    state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
    ts_order = [states[0]['ts'],states[1]['ts']]
    ts_dis = [state_dis[i] for i in ts_order]
    ts2_side = np.sign(ts_dis[1].mean())
    taskinfo['ts2_side'] = ts2_side
    #To ensure TS2 is always associated with the 'top' of the screen, or positive
    #context values, flip the context values if this isn't the case.
    #This ensures that TS1 is always the shape task-set and, for analysis purposes,
    #always associated with the bottom of the screen
    train_dfa['true_context'] = train_dfa['context']
    test_dfa['true_context'] = test_dfa['context']
    
    if ts2_side == -1:
        train_dfa['context'] = train_dfa['context']* -1
        test_dfa['context'] = test_dfa['context']* -1
        ts_dis = ts_dis [::-1]
        
    #What was the mean contextual value for each taskset during this train run?
    train_ts_means = list(train_dfa.groupby('ts').agg(np.mean).context)
    #Same for standard deviation
    train_ts_std = list(train_dfa.groupby('ts').agg(np.std).context)
    train_ts_dis = [norm(m,s) for m,s in zip(train_ts_means,train_ts_std)]
    #And do the same for recursive_p
    train_recursive_p = 1- train_dfa.switch.mean()
    
    
    #decompose contexts
    test_dfa['abs_context'] = abs(test_dfa.context)    
    train_dfa['abs_context'] = abs(train_dfa.context)
    train_dfa['context_sign'] = np.sign(train_dfa.context)
    test_dfa['context_sign'] = np.sign(test_dfa.context)
    #Create vector of context differences
    test_dfa['context_diff'] = test_dfa['context'].diff()
    
    #transform rt
    train_dfa['log_rt'] = np.log(train_dfa.rt)
    test_dfa['log_rt'] = np.log(test_dfa.rt)
    
    #*********************************************
    # Model fitting
    #*********************************************
       
    bias_params = bias_fit_dict[subj_name + '_scalar_cost']
    bias_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = bias_params['tsb'], recursive_prob = bias_params['rp'])
    nobias_params = nobias_fit_dict[subj_name + '_scalar_cost']
    nobias_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = 1, recursive_prob = nobias_params['rp'])
    
    for bias, fit_observer in [('bias',bias_fit_observer), ('nobias', nobias_fit_observer)]:
        #Fit observer for test        
        observer_choices = []
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = trial.context
            posteriors.append(fit_observer.calc_posterior(c)[1])
        posteriors = np.array(posteriors)

        test_dfa[bias + 'fit_observer_posterior'] = posteriors
        test_dfa[bias +'fit_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa[bias +'fit_observer_switch'] = (test_dfa[bias + 'fit_observer_posterior']>.5).diff()
        test_dfa[bias +'conform_fit_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa[bias +'fit_certainty'] = (abs(test_dfa[bias + 'fit_observer_posterior']-.5))/.5
        
        
        #Optimal observer for test        
        optimal_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = 1, recursive_prob = train_recursive_p)
        observer_choices = []
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = trial.context
            posteriors.append(optimal_observer.calc_posterior(c)[1])
        posteriors = np.array(posteriors)
    
        test_dfa['opt_observer_posterior'] = posteriors
        test_dfa['opt_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa['opt_observer_switch'] = (test_dfa.opt_observer_posterior>.5).diff()
        test_dfa['conform_opt_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa['opt_certainty'] = (abs(test_dfa.opt_observer_posterior-.5))/.5
    
    test_dfa['id'] = subj_name
    gtest_df = pd.concat([gtest_df,test_dfa])   
    gtaskinfo.append(taskinfo)
    
gtaskinfo = pd.DataFrame(gtaskinfo)

#Exclude subjects where stim_confom is below some threshold 
select_ids = gtest_df.groupby('id').mean().stim_conform>.75
select_ids = select_ids[select_ids]
select_rows = [i in select_ids for i in gtest_df.id]
gtest_df = gtest_df[select_rows]
ids = select_ids.index

#separate learner group
select_ids = gtest_df.groupby('id').mean().correct > .55
select_ids = select_ids[select_ids]
select_rows = [i in select_ids for i in gtest_df.id]
gtest_learn_df = gtest_df[select_rows]
learn_ids = select_ids.index   
   
   
   
#*********************************************
# Model Comparison
#********************************************* 
compare_df = gtest_learn_df
  
model_subj_compare = compare_df[['subj_ts','opt_observer_posterior','nobiasfit_observer_posterior', 'biasfit_observer_posterior']].corr()

optfit_log_posterior = np.log(abs(compare_df.subj_ts-(1-compare_df.opt_observer_posterior)))
biasfit_log_posterior = np.log(abs(compare_df.subj_ts-(1-compare_df.biasfit_observer_posterior)))
nobiasfit_log_posterior = np.log(abs(compare_df.subj_ts-(1-compare_df.nobiasfit_observer_posterior)))
midline_rule_log_posterior = np.log(abs(compare_df.subj_ts - (1-abs((compare_df.context_sign==1).astype(int)-.1))))

compare_df = pd.concat([compare_df[['id','subj_ts','context']], optfit_log_posterior, biasfit_log_posterior, nobiasfit_log_posterior, midline_rule_log_posterior], axis = 1)
compare_df.columns = ['id','subj_ts','context','optimal','bias','nobias', 'midline']
compare_df['random_log'] = np.log(.5)

summary = compare_df.groupby('id').sum().drop(['context','subj_ts'],axis = 1)
plt.hold(True)
summary.plot(figsize = (16,12), fontsize = 16)
plt.ylabel('Log Posterior')










    
    
    