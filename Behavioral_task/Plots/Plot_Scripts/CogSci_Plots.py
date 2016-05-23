# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:39:58 2016

@author: ian
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd
import seaborn as sns
import warnings
import pickle
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy


# Suppress runtimewarning due to pandas bug
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# *********************************************
# Set up defaults
# *********************************************
plot = True
save = True

# *********************************************
# Load Data
# ********************************************
data_dir = os.path.expanduser('~')
bias2_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/bias2_parameter_fits.p', 'rb'))
bias1_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/bias1_parameter_fits.p', 'rb'))
eoptimal_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/eoptimal_parameter_fits.p', 'rb'))
ignore_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/ignore_parameter_fits.p', 'rb'))
midline_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/midline_parameter_fits.p', 'rb'))
switch_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/switch_parameter_fits.p', 'rb'))
memory_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/memory_parameter_fits.p', 'rb'))
perseverance_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/perseverance_parameter_fits.p', 'rb'))
permem_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/permem_parameter_fits.p', 'rb'))

gtrain_learn_df = pd.read_pickle('../../Analysis/Analysis_Output/gtrain_learn_df.pkl')
gtest_learn_df = pd.read_pickle('../../Analysis/Analysis_Output/gtest_learn_df.pkl')
gtest_conform_df = pd.read_pickle('../../Analysis/Analysis_Output/gtest_conform_df.pkl')
gtest_df = pd.read_pickle('../../Analysis/Analysis_Output/gtest_df.pkl')
gtrain_learn_df.id = gtrain_learn_df.id.astype('str').apply(lambda x: x.zfill(3))
gtest_learn_df.id = gtest_learn_df.id.astype('str').apply(lambda x: x.zfill(3))


# *********************************************
# Select Dataset
# ********************************************* 
model = 'TS'
df = gtest_df.copy()
df.drop(['midline_posterior','midline_posterior_cross'], axis = 1, inplace = True)

# *********************************************
# Additional Variables
# ********************************************* 
for models in ['bias2','bias1','eoptimal', 'ignore', 'switch','memory','perseverance','permem']:
        df[models + '_choice'] = (df[models + '_posterior']>.5).astype(int)
        df[models + '_certainty'] = (abs(df[models + '_posterior']-.5))/.5

switch_sums = []
trials_since_switch = 0
for i,row in df.iterrows():
    if row['switch'] == 1 or pd.isnull(row['switch']):
        trials_since_switch = 0
    else:
        trials_since_switch += 1
    switch_sums.append(trials_since_switch)
df['trials_since_switch'] = switch_sums

df[['last_TS', 'bias2_last_choice']] = df[['subj_ts', 'bias2_choice']].shift(1)
df.loc[0,['last_TS','bias2_last_choice']]=np.nan

# *********************************************
# Selection Criterion
# ********************************************* 
## Exclude subjects based on behavioral criteria
select_ids = gtest_df.groupby('id').mean().stim_conform>.75
#select_ids = np.logical_and(abs(.5-gtest_df.groupby('id')['subj_ts'].mean())<.475, select_ids)
select_ids = select_ids[select_ids]
select_rows = [i in select_ids for i in df.id]
df = df[select_rows]

##exclude based on context sensivitiy
#context_pval = []
#for subj_id in np.unique(df['id']):
#    subj_df = df.query('id == "%s"' % subj_id)
#    res = smf.glm(formula = 'subj_ts ~ context', data = subj_df, family = sm.families.Binomial()).fit()
#    res.summary()
#    context_pval.append(res.pvalues.context)
#select_ids.iloc[:] = np.array(context_pval)<.05
#select_ids = select_ids[select_ids]
#select_rows = [i in select_ids for i in df.id]
#df = df[select_rows]



x = df.groupby('id')['correct'].mean()  
k = range(1,10)
k_error = []
k_track = []
for _ in range(1000):
    for k_i in k:  
        c,label = scipy.cluster.vq.kmeans2(x,k_i)
        k_error.append(np.sum(np.power([c[i] for i in label]-x,2)))
        k_track.append(k_i)
k_df = pd.DataFrame({'k': k_track, 'error': k_error})

#exclude subjects based on percent correct
x = df.groupby('id')['correct'].mean()    
c,label = scipy.cluster.vq.kmeans2(x,[.49,.51])


ids = np.unique(df['id'])
select_ids = ids[label==1]
fail_rows = [i not in select_ids for i in df.id]
select_rows = [i in select_ids for i in df.id]
df_fail = df[fail_rows]
df = df[select_rows]

df.to_csv('../../Analysis/Analysis_Output/learners.csv')
df_fail.to_csv('../../Analysis/Analysis_Output/nonlearners.csv')

# *********************************************
# Model Comparison
# ********************************************* 
compare_df = df.copy()
compare_df_subset = compare_df.filter(regex = 'subj_ts|.*posterior$')
model_subj_compare = compare_df_subset.corr()

log_posteriors = pd.DataFrame()
for model in compare_df_subset.columns[1:]:
    log_posteriors[model] = np.log(abs(compare_df_subset.subj_ts-(1-compare_df_subset[model])))


compare_df = pd.concat([compare_df[['id','subj_ts','context']], log_posteriors], axis = 1)
compare_df['random_log'] = np.log(.5)

summary = compare_df.groupby('id').sum().drop(['context','subj_ts'],axis = 1)

num_params = [3,2,1,1,3,3,4,1,0]
param_cost_df = np.log(df.groupby('id').count()).iloc[:,0:len(summary.columns)]*num_params
param_cost_df.columns = summary.columns
BIC_summary = -2*summary + param_cost_df

#extract column of best model
min_col = BIC_summary.idxmin(1)
best_models = min_col.map(lambda x: x[:x.find('_')])
bayes_models = [x in ['bias2', 'bias1', 'ignore', 'eoptimal'] for x in best_models]
mem_models = [x in ['memory', 'perseverance','permem'] for x in best_models]

best_posterior = []
for i in range(len(best_models)):
    subj_id = best_models.index[i]
    model = best_models[i]
    subj_df = df.query('id == "%s"' % subj_id)
    if model == 'random':
        best_posterior += [.5]*len(subj_df)
    else:
        best_posterior += list(subj_df[model + '_posterior'])
df['best_posterior'] = best_posterior
df['best_choice'] = (df['best_posterior']>.5).astype(int)
df['best_certainty'] = (abs(df['best_posterior']-.5))/.5


# *********************************************
# Behavioral Analysis
# ********************************************* 
#effect of last TS
formula = 'subj_ts ~ context'
delays = list(range(26))
for i in delays[1:]:
    formula += ' + context.shift(%s)' % i

##fit one model across group. Revisit with mixed models
#res = smf.glm(formula = formula, data = df, family = sm.families.Binomial()).fit()
#res.summary()
#learner_params = res.params[1:]
#res = smf.glm(formula = formula, data = df_fail, family = sm.families.Binomial()).fit()
#res.summary()
#nonlearner_params = res.params[1:]
#
#    
learner_params = []
for i in np.unique(df['id']):
    res = smf.glm(formula = formula, data = df.query('id == "%s"' %i), family = sm.families.Binomial()).fit()
    learner_params.append(res.pvalues[1:])
learner_params = pd.DataFrame(learner_params)

select_ids = abs(df_fail.groupby('id').subj_ts.mean()-.5)<.475
select_ids = select_ids[select_ids]
select_rows = [i in select_ids for i in df_fail.id]
df_fail = df_fail[select_rows]
nonlearner_params = []
for i in np.unique(df_fail['id']):
    res = smf.glm(formula = formula, data = df_fail.query('id == "%s"' %i), family = sm.families.Binomial()).fit()
    nonlearner_params.append(res.pvalues[1:])
nonlearner_params = pd.DataFrame(nonlearner_params)


  
# *********************************************
# Plotting
# *********************************************

contexts = np.unique(gtest_df.context)
figdims = (16,12)
fontsize = 30
plot_df = df.copy()
plot_df['rt'] = plot_df['rt']*1000
plot_ids = np.unique(plot_df.id)
if plot == True:
        
    sns.set_style("darkgrid", {"axes.linewidth": "1.25", "axes.edgecolor": ".15"})
    p1 = plt.figure(figsize = figdims)
    plt.hold(True) 
    plt.plot(plot_df.groupby('context').subj_ts.mean(), lw = 4, marker = 'o', markersize = 10, color = 'm', label = 'subject')
    plt.plot(plot_df.groupby('context').bias2_choice.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', label = 'bias-2')
    plt.plot(plot_df.groupby('context').eoptimal_choice.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = 'optimal')
    plt.xticks(list(range(12)),contexts)
    plt.tick_params(labelsize=20)
    plt.xlabel('Stimulus Vertical Position', size = fontsize)
    plt.ylabel('STS choice %', size = fontsize)
    pylab.legend(loc='best',prop={'size':20})
    for subj in plot_ids:
        subj_df = plot_df.query('id == "%s"' %subj)
        if subj_df.correct.mean() < .55:
            plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2, color = 'r', alpha = .2)
        else:
            plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2, color = 'k', alpha = .2)
            
    range_start = 0
    range_length = 5
    a = plt.axes([.60, .15, .35, .35])
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').subj_ts.mean()*100, lw = 2,  alpha = 1, label = 'subject')
    for subj in plot_ids[range_start+1:range_start+range_length]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').subj_ts.mean()*100, lw = 2,  alpha = 1, label = '')
    plt.gca().set_color_cycle(None)
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').bias2_choice.mean()*100, lw = 2, ls = '--', label = 'bias-2')
    for subj in plot_ids[range_start+1:range_start+5]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').bias2_choice.mean()*100, lw = 2, ls = '--', label = '')
    plt.tick_params(
        axis = 'both',
        which = 'both',
        labelleft = 'off',
        labelbottom = 'off')
    pylab.legend(loc='upper left',prop={'size': 20})

    # Plot rt against bias2 model posterior
    sns.set_context('poster')
    subj_df = plot_df.query('rt > 100 & id < "%s"' %plot_ids[3])       
    p2 = sns.lmplot(x='bias2_posterior',y='rt', hue = 'id', data = subj_df, order = 2, size = 6, col = 'id')
    p2.set_xlabels("P(STS)", size = fontsize)
    p2.set_ylabels('Reaction time (ms)', size = fontsize)
    pylab.xlim(0,1)
    
    # Plot rt against bias2 model certainty
    # Take out RT < 100 ms  
    sns.set_context('poster')
    subj_df = plot_df.query('id < "%s"' %plot_ids[3])       
    p3 = sns.lmplot(x ='bias2_certainty', y = 'rt', hue = 'id', col = 'id', size = 8, data = subj_df)   
    p3.set_xlabels("Bias-2 Confidence", size = fontsize)
    p3.set_ylabels('Reaction time (ms)', size = fontsize)
    pylab.xlim(0,1)

    p4 = plt.figure(figsize = figdims)
    plt.hold(True)
    for c in log_posteriors.columns[:-1]:
        sns.kdeplot(summary[c], linewidth = 4)
    plt.legend(loc = 2, fontsize = 20, labels =  [ 'bias-2', 'bias-1', r'optimal', 'base-rate neglect', 'midline'])
    plt.xlabel('Log Likelihood', size = fontsize)
    plt.ylabel('Density (Arbitrary Units)', size = fontsize)
     
    p5 = plt.figure(figsize = figdims)
    p5.subplots_adjust(hspace=.3, wspace = .3)
    
    plt.subplot2grid((2,2),(0,0))
    sns.plt.plot(delays,learner_params.mean(), 'b-o', label = 'Learners', lw = 3)
    sns.plt.plot(delays,nonlearner_params.mean(),'r-o', label = 'Non-Learners', lw = 3)
    plt.xlabel('Context Lag', size = fontsize)
    plt.ylabel('Beta Weights', size = fontsize)
    plt.xlim(-.5,25)
    pylab.legend(loc='best',prop={'size':20})
    plt.tick_params(labelsize=15)
    
    
    plt.subplot2grid((2,2),(1,0), colspan = 1)
    sns.plt.scatter(range(len(x)),x, c = [['r','b'][i] for i in label])
    plt.ylabel('Accuracy', size = fontsize)
    plt.xlabel('Subject Index', size = fontsize)
    plt.xlim([-5,50])
    plt.tick_params(labelsize=15)
    
    plt.subplot2grid((2,2),(1,1), colspan = 1)
    plt.plot(k_df.groupby('k')['error'].mean(),'o-')
    plt.ylabel('SSE', size = fontsize)
    plt.xlabel('Number of Clusters (k)', size = fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot2grid((2,2),(0,1))
    for window in [(0,850)]:
        window_df = df.query('trial_count >= %s and trials_since_switch < 27 and trial_count < %s' % (window[0], window[1]))
        plot_dict = {}
        for i in np.unique(window_df['id']):
            temp_df = window_df.query('id == "%s"' % i)
            plot_dict[i] = [temp_df.query('trials_since_switch == %s' % i)['correct'].mean() for i in np.unique(temp_df['trials_since_switch']) if np.sum(temp_df['trials_since_switch']==i)>5]
            plot_dict['trials_since_switch'] = list(range(max([len(arr) for arr in plot_dict.values()])))
        subplot_df = pd.DataFrame.from_dict(plot_dict, orient='index').transpose()  
        
        subplot_df = pd.melt(subplot_df, id_vars = 'trials_since_switch', var_name = 'id', value_name = 'percent_correct')
        plt.scatter(subplot_df['trials_since_switch'], subplot_df['percent_correct'], color = 'b', alpha = .5)        
    group = window_df.groupby('trials_since_switch').mean()['correct']
    plt.plot(group.index,group,'b-',lw = 4)

    for window in [(0,850)]:
        window_df = df_fail.query('trial_count >= %s and trials_since_switch < 27 and trial_count < %s' % (window[0], window[1]))
        plot_dict = {}
        for i in np.unique(window_df['id']):
            temp_df = window_df.query('id == "%s"' % i)
            plot_dict[i] = [temp_df.query('trials_since_switch == %s' % i)['correct'].mean() for i in np.unique(temp_df['trials_since_switch']) if np.sum(temp_df['trials_since_switch']==i)>5]
            plot_dict['trials_since_switch'] = list(range(max([len(arr) for arr in plot_dict.values()])))
        subplot_df = pd.DataFrame.from_dict(plot_dict, orient='index').transpose()  
        
        subplot_df = pd.melt(subplot_df, id_vars = 'trials_since_switch', var_name = 'id', value_name = 'percent_correct')
        plt.scatter(subplot_df['trials_since_switch'], subplot_df['percent_correct'], color = 'r', alpha = .5)        
    group = window_df.groupby('trials_since_switch').mean()['correct']
    plt.plot(group.index,group,'r-',lw = 4)
    plt.xlim(-1,28) 
    plt.ylim(0,1.1)
    plt.tick_params(labelsize=15)
    plt.ylabel('Percent Correct', size = fontsize)
    plt.xlabel('Trials Since Objective TS Switch', size = fontsize)
    
    if save == True:
        p1.savefig('../CogSci2016/STS%_vs_context.png', format = 'png', dpi = 600, bbox_inches = 'tight')
        p2.savefig('../CogSci2016/rt_vs_posterior_3subj.png',format = 'png', dpi = 600, bbox_inches = 'tight')
        p3.savefig('../CogSci2016/rt_vs_confidence_5subj.png', format = 'png', dpi = 600, bbox_inches = 'tight')
        p4.savefig('../CogSci2016/model_comparison.png', format = 'png', dpi = 600, bbox_inches = 'tight')
        p5.savefig('../CogSci2016/learner_vs_nonlearner.png', format = 'png', dpi = 600, bbox_inches = 'tight')
        
