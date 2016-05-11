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


# Suppress runtimewarning due to pandas bug
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# *********************************************
# Set up defaults
# *********************************************
plot = False
save = True

# *********************************************
# Load Data
# ********************************************
data_dir = os.path.expanduser('~')
bias2_fit_dict = pickle.load(open('Analysis_Output/bias2_parameter_fits.p', 'rb'))
bias1_fit_dict = pickle.load(open('Analysis_Output/bias1_parameter_fits.p', 'rb'))
eoptimal_fit_dict = pickle.load(open('Analysis_Output/eoptimal_parameter_fits.p', 'rb'))
ignore_fit_dict = pickle.load(open('Analysis_Output/ignore_parameter_fits.p', 'rb'))
midline_fit_dict = pickle.load(open('Analysis_Output/midline_parameter_fits.p', 'rb'))
switch_fit_dict = pickle.load(open('Analysis_Output/switch_parameter_fits.p', 'rb'))
memory_fit_dict = pickle.load(open('Analysis_Output/memory_parameter_fits.p', 'rb'))
perseverance_fit_dict = pickle.load(open('Analysis_Output/perseverance_parameter_fits.p', 'rb'))
permem_fit_dict = pickle.load(open('Analysis_Output/permem_parameter_fits.p', 'rb'))



gtrain_learn_df = pd.read_pickle('Analysis_Output/gtrain_learn_df.pkl')
gtest_learn_df = pd.read_pickle('Analysis_Output/gtest_learn_df.pkl')
gtest_conform_df = pd.read_pickle('Analysis_Output/gtest_conform_df.pkl')
gtest_df = pd.read_pickle('Analysis_Output/gtest_df.pkl')
gtrain_learn_df.id = gtrain_learn_df.id.astype('str').apply(lambda x: x.zfill(3))
gtest_learn_df.id = gtest_learn_df.id.astype('str').apply(lambda x: x.zfill(3))


# *********************************************
# Additional Model Variables
# ********************************************* 
model = 'TS'
df = gtest_learn_df.copy()
ids = np.unique(df['id'])
for models in ['bias2','bias1','eoptimal', 'ignore', 'midline', 'switch']:
        df[models + '_choices'] = (df[models + '_posterior']>.5).astype(int)
        df[models + '_certainty'] = (abs(df[models + '_posterior']-.5))/.5
        df[models + '_cross_choices'] = (df[models + '_posterior_cross']>.5).astype(int)
        df[models + '_cross_certainty'] = (abs(df[models + '_posterior_cross']-.5))/.5
            
# *********************************************
# Calculate how well the bias-2 model does
# ********************************************* 
r1 = pd.concat([df['id'], df['subj_ts']==df['bias1_choices']],1)
np.mean(r1.groupby('id').mean())

np.mean([bias2_fit_dict[w + '_' + model + '_fullRun']['r2'] for w in ids] + [bias2_fit_dict[w + '_' + model + '_fullRun']['r1'] for w in ids])
# *********************************************
# Model Comparison
# ********************************************* 
compare_df = gtest_learn_df
compare_df_subset = compare_df.filter(regex = 'subj_ts|.*posterior_cross$')
model_subj_compare = compare_df_subset.corr()

log_posteriors = pd.DataFrame()
for model in compare_df_subset.columns[1:]:
    log_posteriors[model] = np.log(abs(compare_df_subset.subj_ts-(1-compare_df_subset[model])))


compare_df = pd.concat([compare_df[['id','subj_ts','context']], log_posteriors], axis = 1)
compare_df['random_log'] = np.log(.5)

summary = compare_df.groupby('id').sum().drop(['context','subj_ts'],axis = 1)

# *********************************************
# Behavioral Analysis
# ********************************************* 

switch_sums = []
trials_since_switch = 0
for i,row in df.iterrows():
    if row['switch'] == 1 or pd.isnull(row['switch']):
        trials_since_switch = 0
    else:
        trials_since_switch += 1
    switch_sums.append(trials_since_switch)
df['trials_since_switch'] = switch_sums

res = sm.OLS(df['correct'], df[['trials_since_switch']]).fit()
print(res.summary())

#df['model_correct'] = df['bias2_choices']==df['ts']
#res = sm.OLS(df['model_correct'], df[['trials_since_switch']]).fit()
#print(res.summary())

for window in [(0,850)]:
    window_df = df.query('trial_count >= %s and trial_count < %s' % (window[0], window[1]))
    plot_dict = {}
    for i in np.unique(window_df['id']):
        temp_df = window_df.query('id == "%s"' % i)
        plot_dict[i] = [temp_df.query('trials_since_switch == %s' % i)['correct'].mean() for i in np.unique(temp_df['trials_since_switch'])]
        plot_dict['trials_since_switch'] = list(range(max([len(arr) for arr in plot_dict.values()])))
    plot_df = pd.DataFrame.from_dict(plot_dict, orient='index').transpose()  
    
    plot_df = pd.melt(plot_df, id_vars = 'trials_since_switch', var_name = 'id', value_name = 'percent_correct')
    sns.lmplot(x = 'trials_since_switch', y = 'percent_correct', data = plot_df, size = 8, hue = 'id', fit_reg = False)
    plt.title('Trial window: ' + str(window), size = 20)
    

# *********************************************
# Plotting
# *********************************************

contexts = np.unique(gtest_df.context)
figdims = (16,12)
fontsize = 20
plot_df = gtest_learn_df.copy()
plot_df['rt'] = plot_df['rt']*1000
plot_ids = np.unique(plot_df.id)
if plot == True:
    
    # Plot task-set count by context value
    sns.set_style("darkgrid", {"axes.linewidth": "1.25", "axes.edgecolor": ".15"})
    p1 = plt.figure(figsize = figdims)
    plt.hold(True) 
    plt.plot(plot_df.groupby('context').subj_ts.mean(), lw = 4, marker = 'o', markersize = 10, color = 'm', label = 'subject')
    plt.plot(plot_df.groupby('context').bias2_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', label = 'bias-2 observer')
    plt.plot(plot_df.groupby('context').bias1_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = 'bias-1 observer')
    plt.xticks(list(range(12)),contexts)
    plt.xlabel('Stimulus Vertical Position', size = fontsize)
    plt.ylabel('TS2 choice %', size = fontsize)
    pylab.legend(loc='best',prop={'size':20})
    for subj in plot_ids:
        subj_df = plot_df.query('id == "%s"' %subj)
        if subj_df.correct.mean() < .55:
            plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2, color = 'r', alpha = .2)
        else:
            plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2, color = 'k', alpha = .2)
    a = plt.axes([.62, .15, .3, .3])
    plt.plot(plot_df.groupby('context').subj_ts.mean(), lw = 4, marker = 'o', markersize = 10, color = 'm', label = 'subject')
    plt.plot(plot_df.groupby('context').eoptimal_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = r'$\epsilon$-optimal observer')
    plt.plot(plot_df.groupby('context').midline_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = ':', label = 'midline rule')
    plt.tick_params(
        axis = 'both',
        which = 'both',
        labelleft = 'off',
        labelbottom = 'off')
    pylab.legend(loc='upper left',prop={'size':14})
    

    # Plot task-set count by context value
    range_start = 0
    p2 = plt.figure(figsize = figdims)
    plt.hold(True) 
    plt.xticks(list(range(12)),contexts)
    plt.xlabel('Stimulus Vertical Position', size = fontsize)
    plt.ylabel('STS choice %', size = fontsize)
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2,  alpha = 1, label = 'subject')
    for subj in plot_ids[range_start+1:range_start+5]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2,  alpha = 1, label = '_nolegend_')
    plt.gca().set_color_cycle(None)
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').bias2_choices.mean(), lw = 2, ls = '--', label = 'bias-2 observer')
    for subj in plot_ids[range_start+1:range_start+5]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').bias2_choices.mean(), lw = 2, ls = '--', label = '_nolegend_')
    pylab.legend(loc='best',prop={'size':20})

    # look at RT
    p4 = plt.figure(figsize = figdims)
    plt.subplot(4,1,1)
    plot_df.rt.hist(bins = 25)
    plt.ylabel('Frequency', size = fontsize)
    
    plt.subplot(4,1,2)    
    plt.hold(True)
    sns.kdeplot(plot_df.query('subj_switch == 0')['rt'],color = 'm', lw = 5, label = 'stay')
    sns.kdeplot(plot_df.query('subj_switch == 1')['rt'],color = 'c', lw = 5, label = 'switch')
    plot_df.query('subj_switch == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    plot_df.query('subj_switch == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    pylab.legend(loc='upper right',prop={'size':20})
    plt.xlim(xmin=0)

    
    plt.subplot(4,1,3)
    plt.hold(True)
    sns.kdeplot(plot_df.query('subj_switch == 0 and rep_resp == 1')['rt'], color = 'm', lw = 5, label = 'repeat response')
    sns.kdeplot(plot_df.query('subj_switch == 0 and rep_resp == 0')['rt'], color = 'c', lw = 5, label = 'change response (within task-set)')
    plot_df.query('subj_switch == 0 and rep_resp == 1')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    plot_df.query('subj_switch == 0 and rep_resp == 0')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.ylabel('Probability Density', size = fontsize)
    pylab.legend(loc='upper right',prop={'size':20})
    plt.xlim(xmin=0)

        
    plt.subplot(4,1,4)
    plt.hold(True)
    sns.kdeplot(plot_df.query('subj_ts == 0')['rt'], color = 'm', lw = 5, label = 'ts1')
    sns.kdeplot(plot_df.query('subj_ts == 1')['rt'], color = 'c', lw = 5, label = 'ts2')
    plot_df.query('subj_ts == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    plot_df.query('subj_ts == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('Reaction Time (ms)', size = fontsize)
    pylab.legend(loc='upper right',prop={'size':20})
    plt.xlim(xmin=0)

    	    
    # RT for switch vs stay for different trial-by-trial context diff
    p5 = plot_df.groupby(['subj_switch','context_diff']).mean().rt.unstack(level = 0).plot(marker = 'o',color = ['c','m'], figsize = figdims, fontsize = fontsize)     
    p5 = p5.get_figure()
    
    # Plot rt against bias2 model posterior
    sns.set_context('poster')
    subj_df = plot_df.query('rt > 100 & id < "%s"' %plot_ids[3])       
    p6 = sns.lmplot(x='bias2_posterior',y='rt', hue = 'id', data = subj_df, order = 2, size = 6, col = 'id')
    p6.set_xlabels("P(TS2)", size = fontsize)
    p6.set_ylabels('Reaction time (ms)', size = fontsize)
    
    # Plot rt against bias2 model certainty
    # Take out RT < 100 ms  
    sns.set_context('poster')
    subj_df = plot_df.query('rt > 100 & id < "%s"' %plot_ids[3])       
    p7 = sns.lmplot(x ='bias2_certainty', y = 'rt', hue = 'id', col = 'id', size = 6, data = subj_df)   
    p7.set_xlabels("Model Confidence", size = fontsize)
    p7.set_ylabels('Reaction time (ms)', size = fontsize)
    
    p8 = sns.lmplot(x ='bias2_certainty', y = 'rt', hue = 'id', ci = None, legend = False, size = figdims[1], data = plot_df.query('rt>100'))  
    plt.xlim(-.1,1.1)
    p8.set_xlabels("Model Confidence", size = fontsize)
    p8.set_ylabels('Reaction time (ms)', size = fontsize)
    
    # plot bias2 parameters
    params_df = pd.DataFrame()
    params_df['id'] = [x[1:3] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df['learner'] = [x[0:3] in plot_ids for x in bias2_fit_dict if ('_fullRun' in x)] 
    params_df['r1'] = [bias2_fit_dict[x]['r1'] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df['r2'] = [bias2_fit_dict[x]['r2'] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df['eps'] = [bias2_fit_dict[x]['eps'] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df = pd.melt(params_df, id_vars = ['id','learner'], value_vars = ['eps','r1','r2'], var_name = 'param', value_name = 'val')

    p9 = plt.figure(figsize = figdims)
    box_palette = sns.color_palette(['m','c'], desat = 1)
    sns.boxplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, palette = box_palette)
    sns.stripplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, jitter = True, edgecolor = "gray", palette = box_palette)
    plt.xlabel("Parameter", size = fontsize)
    plt.ylabel('Value', size = fontsize)
    plt.title('Bias-2 Model Parameter Fits', size = fontsize+4)
    plt.xticks([0,1,2], ('$\epsilon$','$r_1$','$r_2$'), size = fontsize)
    np.corrcoef(gtest_learn_df.rt,gtest_learn_df.bias2_posterior)
    
    # plot bias1 parameters
    params_df = pd.DataFrame()
    params_df['id'] = [x[1:3] for x in bias1_fit_dict if ('_fullRun' in x)]
    params_df['learner'] = [x[0:3] in plot_ids for x in bias1_fit_dict if ('_fullRun' in x)] 
    params_df['r1'] = [bias2_fit_dict[x]['r1'] for x in bias1_fit_dict if ('_fullRun' in x)]
    params_df['eps'] = [bias2_fit_dict[x]['eps'] for x in bias1_fit_dict if ('_fullRun' in x)]
    params_df = pd.melt(params_df, id_vars = ['id','learner'], value_vars = ['eps','r1'], var_name = 'param', value_name = 'val')

    p10 = plt.figure(figsize = figdims)
    box_palette = sns.color_palette(['m','c'], desat = 1)
    sns.boxplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, palette = box_palette)
    sns.stripplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, jitter = True, edgecolor = "gray", palette = box_palette)
    plt.xlabel("Parameter", size = fontsize)
    plt.ylabel('Value', size = fontsize)
    plt.title('Bias-1 Model Parameter Fits', size = fontsize+4)
    plt.xticks([0,1,2], ('$\epsilon$','$r_1$'), size = fontsize)
    np.corrcoef(gtest_learn_df.rt,gtest_learn_df.bias2_posterior)

    #look at models
    p11 = plt.figure(figsize = figdims)
    plt.hold(True)
    for c in log_posteriors.columns[:-1]:
        sns.kdeplot(summary[c])
    
    p12 = sns.heatmap(model_subj_compare)
    p13 = sns.heatmap(df.filter(regex='choices|subj_ts').corr())
    if save == True:
        p1.savefig('../Plots/TS2%_vs_context.png', format = 'png', dpi = 300)
        p2.savefig('../Plots/Individual_subject_fits.png',format = 'png', dpi = 300)
        p3.savefig('../Plots/TS_proportions.png', format = 'png', dpi = 300)
        p4.savefig('../Plots/RTs.png', format = 'png')
        p5.savefig('../Plots/RT_across_context_diffs.png', format = 'png', dpi = 300)
        p6.savefig('../Plots/rt_vs_posterior_3subj.png', format = 'png', dpi = 300)
        p7.savefig('../Plots/rt_vs_confidence_3subj.png', format = 'png', dpi = 300)
        p8.savefig('../Plots/rt_vs_confidence.png', format = 'png', dpi = 300)
        p9.savefig('../Plots/bias2_param_value.png', format = 'png', dpi = 300)
        p10.savefig('../Plots/bias1_param_value.png', format = 'png', dpi = 300)
        p11.savefig('../Plots/model_comparison.png', format = 'png', dpi = 300)
        