# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:12:43 2015

@author: Ian
"""


import numpy as np
from scipy.stats import norm
from Load_Data import load_data
from helper_classes import BiasPredModel, SwitchModel
import pickle, glob, re, lmfit, os
import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd
import seaborn as sns
from collections import OrderedDict as odict
import warnings



#*********************************************
# Set up plotting defaultsfrom helper_functions import calc_posterior, plot_run

#*********************************************

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20,
        }
        
axes = {'titleweight' : 'bold'
        }
plt.rc('font', **font)
plt.rc('axes', **axes)

save = False
plot = False
fitting = True

#*********************************************
# Load Data
#*********************************************
data_dir = "D:\\Ian"
data_dir = "/mnt/Data/Ian"
try:
    bias2_fit_dict = pickle.load(open('Analysis_Output/bias2_parameter_fits.p', 'rb'))
except:
    bias2_fit_dict = {}
try:
    bias1_fit_dict = pickle.load(open('Analysis_Output/bias1_parameter_fits.p', 'rb'))
except:
    bias1_fit_dict = {}
try:
    eoptimal_fit_dict = pickle.load(open('Analysis_Output/eoptimal_parameter_fits.p', 'rb'))
except:
    eoptimal_fit_dict = {}
try:
    ignore_fit_dict = pickle.load(open('Analysis_Output/ignore_parameter_fits.p', 'rb'))
except:
    ignore_fit_dict = {}
try:
    midline_fit_dict = pickle.load(open('Analysis_Output/midline_parameter_fits.p', 'rb'))
except:
    midline_fit_dict = {}
try:
    switch_fit_dict = pickle.load(open('Analysis_Output/switch_parameter_fits.p', 'rb'))
except:
    switch_fit_dict = {}
    
    
train_files = glob.glob(data_dir + '/Mega/IanE_RawData/Prob_Context_Task/RawData/*Context_20*yaml')
test_files = glob.glob(data_dir + '/Mega/IanE_RawData/Prob_Context_Task/RawData/*Context_test*yaml')

gtest_learn_df = pd.DataFrame.from_csv('Analysis_Output/gtest_learn_df.csv')
gtest_df = pd.DataFrame.from_csv('Analysis_Output/gtest_df.csv')
subj = 30
test_dfa = gtest_df[gtest_df['id'] == subj]
#*********************************************
# Generic Experimental Settings
#*********************************************

behav_sum['train_len'] = len(train_dfa)
behav_sum['test_len'] = len(test_dfa)

#*********************************************
# Performance
#*********************************************    

#accuracy is defined in relation to the ideal observer
behav_sum['train_ts1_acc'], behav_sum['train_ts2_acc'] = list(train_dfa.groupby('ts').conform_opt_observer.mean())
behav_sum['test_ts1_acc'], behav_sum['test_ts2_acc'] = list(test_dfa.groupby('ts').conform_opt_observer.mean())

#Very course estimate of learning: is there a change in performance over trials?
#Threshold p < .01, and if so, what direction?
learn_direct = []
for sub in [train_dfa, test_dfa]:
    logit = sm.Logit(sub['conform_opt_observer'], sm.add_constant(sub[['trial_count']]))
    result = logit.fit()
    learn_direct.append(int(result.pvalues[1]<.01) * np.sign(result.params[1]))
behav_sum['learning?'] = learn_direct

behav_sum['TS2_percent'] = test_dfa.groupby('context').subj_ts.mean()

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
# Switch Analysis
#*********************************************
#Count the number of times there was a switch to each TS for each context value
switch_counts = odict()
switch_counts['ignore_observer'] = test_dfa.query('ignore_observer_switch == True').groupby(['ignore_observer_choices','context']).trial_count.count().unstack(level = 0)
switch_counts['subject'] = test_dfa.query('subj_switch == True').groupby(['subj_ts','context']).trial_count.count().unstack(level = 0)
switch_counts['opt_observer'] = test_dfa.query('opt_observer_switch == True').groupby(['opt_observer_choices','context']).trial_count.count().unstack(level = 0)
try:
    switch_counts['fit_observer'] = test_dfa.query('fit_observer_switch == True').groupby(['fit_observer_choices','context']).trial_count.count().unstack(level = 0)
except:
    print("No fit observer!")

#normalize switch counts by the ignore rule. The ignore rule represents
#the  number of switches someone would make if they switched task-sets
#every time the stimuli's position crossed the ignore to that position
norm_switch_counts = odict()
for key in switch_counts:
    empty_df = pd.DataFrame(index = np.unique(test_dfa.context), columns = [0,1])
    empty_df.index.name = 'context'
    empty_df.loc[switch_counts[key].index] = switch_counts[key]
    switch_counts[key] = empty_df
    norm_switch_counts[key] = switch_counts[key].div(switch_counts['ignore_observer'],axis = 0)

behav_sum['switch_counts'] = switch_counts['subject']
behav_sum['ts2_side'] = ts2_side
behav_sum['norm_switch_counts'] = norm_switch_counts['subject']

#*********************************************
# linear fit of RT based on different factors
#*********************************************

#absolute context
result = sm.GLS(np.log(test_dfa.rt),sm.add_constant(test_dfa.abs_context)).fit()
behav_sum['context->rt'] = result.params[1] * int(result.pvalues[1]<.05)

#optimal model confidence
result = sm.GLS(np.log(test_dfa.rt),sm.add_constant(test_dfa.opt_certainty)).fit()
print(result.summary())

try:
    result = sm.GLS(np.log(test_dfa.rt),sm.add_constant(test_dfa.fit_certainty)).fit()
    print(result.summary())
except:
    print("No fit observer!")
    

#*********************************************
# Models
#*********************************************

model_subj_compare = test_dfa[['subj_ts','fit_observer_posterior', 'opt_observer_posterior', 'ignore_observer_posterior']].corr()

fit_log_posterior = np.sum(np.log([abs(test_dfa.subj_ts.loc[i] - (1-test_dfa.opt_observer_posterior.loc[i])) for i in test_dfa.index]))
midline_rule_log_posterior = np.sum(np.log([abs(test_dfa.subj_ts.loc[i] - (1-abs(test_dfa.ignore_observer_choices.loc[i]-.2))) for i in test_dfa.index]))


#*********************************************
# Plotting
#*********************************************


if plot == True:
    contexts = np.unique(test_dfa.context)
    figdims = (16,12)
    fontsize = 20
    
    plotting_dict = odict()
    plotting_dict['ignore'] = ['ignore_observer_posterior', 'r','base rate neglect']
    plotting_dict['eoptimal'] = ['eoptimal_observer_posterior', 'm','eoptimal']
    plotting_dict['bias2'] = ['bias2_observer_posterior', 'c','bias2']
    
    #bias2 much better fit than optimal, incorporates bias
    sub_id = '041'
    sub = gtest_df[gtest_df['id'] == int(sub_id)]
    sub = sub[250:400]
    p1 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, exclude = ['bias2','eoptimal'], fontsize = 20)
    p1.savefig('../Plots/Subj_Plots/' + sub_id + '_1.png', format = 'png', dpi = 300)
    p2 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, exclude = ['bias2'], fontsize = 20)
    p2.savefig('../Plots/Subj_Plots/' + sub_id + '_2.png', format = 'png', dpi = 300)
    p3 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, fontsize = 20)
    p3.savefig('../Plots/Subj_Plots/' + sub_id + '_3.png', format = 'png', dpi = 300)
    
    #equivalent fit
    sub_id = '058'
    sub = gtest_df[gtest_df['id'] == int(sub_id)]
    sub = sub[250:400]
    p1 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, exclude = ['bias2','eoptimal'], fontsize = 20)
    p1.savefig('../Plots/Subj_Plots/' + sub_id + '_1.png', format = 'png', dpi = 300)
    p2 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, exclude = ['bias2'], fontsize = 20)
    p2.savefig('../Plots/Subj_Plots/' + sub_id + '_2.png', format = 'png', dpi = 300)
    p3 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, fontsize = 20)
    p3.savefig('../Plots/Subj_Plots/' + sub_id + '_3.png', format = 'png', dpi = 300)
    
    #bias2 better fit, more extreme values
    sub_id = '085'
    sub = gtest_df[gtest_df['id'] == int(sub_id)]
    sub = sub[250:400]
    p1 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, exclude = ['bias2','eoptimal'], fontsize = 20)
    p1.savefig('../Plots/Subj_Plots/' + sub_id + '_1.png', format = 'png', dpi = 300)
    p2 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, exclude = ['bias2'], fontsize = 20)
    p2.savefig('../Plots/Subj_Plots/' + sub_id + '_2.png', format = 'png', dpi = 300)
    p3 = plt.figure(figsize = figdims)
    plot_run(sub,plotting_dict, fontsize = 20)
    p3.savefig('../Plots/Subj_Plots/' + sub_id + '_3.png', format = 'png', dpi = 300)
    





