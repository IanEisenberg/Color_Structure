"""
Created on Mon Apr 27 11:16:08 2015

@author: Ian
"""
import sys
sys.path.insert(0, '../../Analysis/')

import numpy as np
from scipy.stats import norm
from Load_Data import load_data
from helper_classes import BiasPredModel, SwitchModel
from helper_functions import calc_posterior
import pickle, glob, re, lmfit, os
import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd
import seaborn as sns
from collections import OrderedDict as odict
import warnings

# Suppress runtimewarning due to pandas bug
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# *********************************************
# Set up defaults
# *********************************************

plot = True
save = False

# *********************************************
# Load Data
# ********************************************
home = os.path.expanduser("~")
try:
    bias2_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/bias2_parameter_fits.p', 'rb'))
except:
    bias2_fit_dict = {}
try:
    bias1_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/bias1_parameter_fits.p', 'rb'))
except:
    bias1_fit_dict = {}
try:
    eoptimal_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/eoptimal_parameter_fits.p', 'rb'))
except:
    eoptimal_fit_dict = {}
try:
    ignore_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/ignore_parameter_fits.p', 'rb'))
except:
    ignore_fit_dict = {}
try:
    midline_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/midline_parameter_fits.p', 'rb'))
except:
    midline_fit_dict = {}
try:
    switch_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/switch_parameter_fits.p', 'rb'))
except:
    switch_fit_dict = {}

gtest_learn_df_crossval = pd.DataFrame.from_csv('../../Analysis/Analysis_Output/gtest_learn_df_crossval.csv')
gtest_learn_df = pd.DataFrame.from_csv('../../Analysis/Analysis_Output/gtest_learn_df.csv')
gtest_df = pd.DataFrame.from_csv('../../Analysis/Analysis_Output/gtest_df.csv')
gtest_learn_df.id = gtest_learn_df.id.astype('str')
gtest_df.id = gtest_df.id.astype('str')
gtest_learn_df.id = gtest_learn_df.id.apply(lambda x: x.zfill(3))
gtest_df.id = gtest_df.id.apply(lambda x: x.zfill(3))


# *********************************************
# Model Comparison
# ********************************************* 
compare_df = gtest_learn_df_crossval
compare_df_subset= compare_df[['subj_ts','bias2_observer_posterior','bias1_observer_posterior','eoptimal_observer_posterior','ignore_observer_posterior','midline_observer_posterior','switch_observer_posterior']]
model_subj_compare = compare_df_subset.corr()

log_posteriors = pd.DataFrame()
for model in compare_df_subset.columns[1:]:
    log_posteriors[model] = np.log(abs(compare_df_subset.subj_ts-(1-compare_df_subset[model])))


compare_df = pd.concat([compare_df[['id','subj_ts','context']], log_posteriors], axis = 1)
compare_df['random_log'] = np.log(.5)

summary = compare_df.groupby('id').sum().drop(['context','subj_ts'],axis = 1)


  
# *********************************************
# Plotting
# *********************************************

contexts = np.unique(gtest_df.context)
figdims = (16,12)
fontsize = 30
plot_df = gtest_learn_df.copy()
plot_df['rt'] = plot_df['rt']*1000
plot_ids = np.unique(plot_df.id)
if plot == True:
        
     # Plot task-set count by context value
    sns.set_style("darkgrid", {"axes.linewidth": "1.25", "axes.edgecolor": ".15"})
    p1 = plt.figure(figsize = figdims)
    plt.hold(True) 
    plt.plot(plot_df.groupby('context').subj_ts.mean()*100, lw = 4, marker = 'o', markersize = 10, color = 'm', label = 'subject')
    plt.plot(plot_df.groupby('context').bias2_observer_choices.mean()*100, lw = 4, marker = 'o', markersize = 10, color = 'c', label = 'bias-2')
    plt.plot(plot_df.groupby('context').eoptimal_observer_choices.mean()*100, lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = r'optimal')
    plt.xlabel('Stimulus Vertical Position', size = fontsize)
    plt.ylabel('STS choice %', size = fontsize)
    pylab.legend(loc='best',prop={'size': 24})
    for subj in plot_ids:
        subj_df = plot_df.query('id == "%s"' %subj)
        if subj_df.correct.mean() < .55:
            plt.plot(subj_df.groupby('context').subj_ts.mean()*100, lw = 2, color = 'r', alpha = .2)
        else:
            plt.plot(subj_df.groupby('context').subj_ts.mean()*100, lw = 2, color = 'k', alpha = .2)
    a = plt.axes([.60, .15, .35, .35])
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').subj_ts.mean()*100, lw = 2,  alpha = 1, label = 'subject')
    for subj in plot_ids[range_start+1:range_start+5]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').subj_ts.mean()*100, lw = 2,  alpha = 1, label = '')
    plt.gca().set_color_cycle(None)
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').bias2_observer_choices.mean()*100, lw = 2, ls = '--', label = 'bias-2')
    for subj in plot_ids[range_start+1:range_start+5]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').bias2_observer_choices.mean()*100, lw = 2, ls = '--', label = '')
    plt.tick_params(
        axis = 'both',
        which = 'both',
        labelleft = 'off',
        labelbottom = 'off')
    pylab.legend(loc='upper left',prop={'size': 20})

    # Plot rt against bias2 model posterior
    sns.set_context('poster')
    subj_df = plot_df.query('rt > 100 & id < "%s"' %plot_ids[3])       
    p2 = sns.lmplot(x='bias2_observer_posterior',y='rt', hue = 'id', data = subj_df, order = 2, size = 6, col = 'id')
    p2.set_xlabels("P(STS)", size = fontsize)
    p2.set_ylabels('Reaction time (ms)', size = fontsize)
    pylab.xlim(0,1)
    
    # Plot rt against bias2 model certainty
    # Take out RT < 100 ms  
    sns.set_context('poster')
    subj_df = plot_df.query('id < "%s"' %plot_ids[5])       
    p3 = sns.lmplot(x ='bias2_certainty', y = 'rt', hue = 'id', col = 'id', size = 6, data = subj_df)   
    p3.set_xlabels("Model Confidence", size = fontsize)
    p3.set_ylabels('Reaction time (ms)', size = fontsize)
    pylab.xlim(0,1)

    p4 = plt.figure(figsize = figdims)
    plt.hold(True)
    for c in log_posteriors.columns[:-1]:
        sns.kdeplot(summary[c], linewidth = 4)
    plt.legend(loc = 2, fontsize = 20, labels =  [ 'bias-2', 'bias-1', r'optimal', 'base-rate neglect', 'midline'])
    plt.xlabel('Log Likelihood', size = fontsize)
    plt.ylabel('Density (Arbitrary Units)', size = fontsize)
      
    if save == True:
        p1.savefig('../CogSci2016/STS%_vs_context.png', format = 'png', dpi = 600, bbox_inches = 'tight')
        p2.savefig('../CogSci2016/rt_vs_posterior_3subj.png',format = 'png', dpi = 600, bbox_inches = 'tight')
        p3.savefig('../CogSci2016/rt_vs_confidence_5subj.png', format = 'png', dpi = 600, bbox_inches = 'tight')
        p4.savefig('../CogSci2016/model_comparison.png', format = 'png', dpi = 600, bbox_inches = 'tight')
        
