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
plot = False
save = False
print_diagnoistic = True

# *********************************************
# Load Data
# ********************************************
data_dir = os.path.expanduser('~')
bias2_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/bias2_parameter_fits.pkl', 'rb'))
bias1_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/bias1_parameter_fits.pkl', 'rb'))
eoptimal_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/eoptimal_parameter_fits.pkl', 'rb'))
ignore_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/ignore_parameter_fits.pkl', 'rb'))
midline_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/midline_parameter_fits.pkl', 'rb'))
switch_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/switch_parameter_fits.pkl', 'rb'))
memory_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/memory_parameter_fits.pkl', 'rb'))
perseverance_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/perseverance_parameter_fits.pkl', 'rb'))
permem_fit_dict = pickle.load(open('../../Analysis/Analysis_Output/permem_parameter_fits.pkl', 'rb'))

gtrain_df = pd.read_pickle('../../Analysis/Analysis_Output/gtrain_df.pkl')
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
if 'midline_posterior' in df.columns:
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


# *********************************************
# Selection Criterion
# ********************************************* 
## Exclude subjects based on behavioral criteria
select_ids = gtest_df.groupby('id').mean().stim_conform>.75
select_ids = list(select_ids[select_ids].index)
utter_failures = df.query('id not in %s' % select_ids)
df = df.query('id in %s' % select_ids)



group_means = df.groupby('id')['correct'].mean()  
k = range(1,10)
k_error = []
for k_i in k:  
    c,label = scipy.cluster.vq.kmeans2(group_means,k_i)
    k_error.append(np.sum(np.power([c[i] for i in label]-group_means,2)))

#exclude subjects based on percent correct
x = df.groupby('id')['correct'].mean()    
c,label = scipy.cluster.vq.kmeans2(group_means,np.array([.49,.51]))



# *********************************************
# Model Comparison
# ********************************************* 
compare_df = df
compare_df_subset = compare_df.filter(regex = 'subj_ts|.*posterior_cross$')
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
bayes_models = [i in ['bias2', 'bias1', 'ignore', 'eoptimal'] for i in best_models]
mem_models = [i in ['memory', 'perseverance','permem'] for i in best_models]

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

all_df = df.copy()
ids = np.unique(df['id'])
select_ids = list(ids[label==1])
df_nonlearners = df.query('id not in %s' % select_ids)
df = df.query('id in %s' % select_ids)

# *********************************************
# Behavioral Analysis
# ********************************************* 
#train analysis
gtrain_df.loc[:,'last_FB'] = gtrain_df['FB'].shift()
gtrain_df.loc[:,'last_choice'] = gtrain_df['subj_ts'].shift()
gtrain_df.loc[df.index == 0,['last_FB','last_choice']] = np.nan

params = []
pvals = []
formula = 'subj_ts ~ context + last_choice * last_FB'
delays = list(range(26))
for i in delays[1:]:
    formula += ' + context.shift(%s)' % i
for i in np.unique(df['id']):
    res = smf.glm(formula = formula, data = gtrain_df.query('id == "%s"' %i), family = sm.families.Binomial()).fit()
    params.append(res.params[1:])    
    pvals.append(res.pvalues[1:])


#effect of last TS
df[['last_TS', 'bias2_last_choice']] = df[['subj_ts', 'bias2_choice']].shift(1)
df.loc[0,['last_TS','bias2_last_choice']]=np.nan
formula = 'subj_ts ~ context'
delays = list(range(26))
for i in delays[1:]:
    formula += ' + context.shift(%s)' % i


learner_params = []
for i in np.unique(df['id']):
    res = smf.glm(formula = formula, data = df.query('id == "%s"' %i), family = sm.families.Binomial()).fit()
    learner_params.append(res.params[1:])
learner_params = pd.DataFrame(learner_params).mean()

select_ids = abs(df_nonlearners.groupby('id').subj_ts.mean()-.5)<.475
select_ids = list(select_ids[select_ids].index)
df_fail = df_nonlearners.query('id in %s' % select_ids)
nonlearner_params = []
for i in np.unique(df_fail['id']):
    res = smf.glm(formula = formula, data = df_fail.query('id == "%s"' %i), family = sm.families.Binomial()).fit()
    nonlearner_params.append(res.params[1:])
nonlearner_params = pd.DataFrame(nonlearner_params).mean()

# *********************************************
# Print Diagnostics
# *********************************************

if print_diagnoistic == True:
    for i in all_df.id.unique():
        id_df = all_df[all_df['id'] == i]
        if i in df.id.unique():
            print(i, 'learner, best: ', best_models.loc[i])
        else:
            print(i, 'nonlearner, best: ',  best_models.loc[i])
        print('Pereseverance:',  np.corrcoef(id_df.subj_ts,id_df.perseverance_choice)[1,0]) 
        print('bias2:',  np.corrcoef(id_df.subj_ts,id_df.bias2_choice)[1,0])
        print('bias1:',  np.corrcoef(id_df.subj_ts,id_df.bias1_choice)[1,0]) 
        print('')


# *********************************************
# Plotting
# *********************************************

if plot == True:
    contexts = np.unique(gtest_df.context)
    figdims = (16,12)
    fontsize = 20
    
    # ***************************
    # Plots for Learners - only using bayesian models
    # ***************************
    plot_df = df.copy()
    plot_df['rt'] = plot_df['rt']*1000
    plot_ids = np.unique(plot_df.id)
    
    # Plot task-set count by context value
    sns.set_style("darkgrid", {"axes.linewidth": "1.25", "axes.edgecolor": ".15"})
    p1 = plt.figure(figsize = figdims)
    plt.hold(True) 
    plt.plot(plot_df.groupby('context').subj_ts.mean(), lw = 4, marker = 'o', markersize = 10, color = 'm', label = 'subject')
    plt.plot(plot_df.groupby('context').bias2_choice.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', label = 'bias-2 observer')
    plt.plot(plot_df.groupby('context').bias1_choice.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = 'bias-1 observer')
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
    plt.plot(plot_df.groupby('context').eoptimal_choice.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = r'$\epsilon$-optimal observer')
    plt.tick_params(
        axis = 'both',
        which = 'both',
        labelleft = 'off',
        labelbottom = 'off')
    pylab.legend(loc='upper left',prop={'size':14})
    

    # Plot task-set count by context value
    range_start = 0
    range_length = 7
    p2 = plt.figure(figsize = figdims)
    plt.hold(True) 
    plt.xticks(list(range(12)),contexts)
    plt.xlabel('Stimulus Vertical Position', size = fontsize)
    plt.ylabel('TS2 choice %', size = fontsize)
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2,  alpha = 1, label = 'subject')
    for subj in plot_ids[range_start+1:range_start+range_length]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2,  alpha = 1, label = '_nolegend_')
    plt.gca().set_color_cycle(None)
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').bias2_choice.mean(), lw = 2, ls = '--', label = 'bias-2 observer')
    for subj in plot_ids[range_start+1:range_start+range_length]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').bias2_choice.mean(), lw = 2, ls = '--', label = '_nolegend_')
    pylab.legend(loc='best',prop={'size':20})

    
    # Plot rt against bias2 model posterior
    sns.set_context('poster')
    subj_df = plot_df.query('rt > 100 & id < "%s"' %plot_ids[4])       
    p3 = sns.lmplot(x='bias2_posterior',y='rt', hue = 'id', data = subj_df, order = 2, size = 6, col = 'id')
    p3.set_xlabels("P(TS2)", size = fontsize)
    p3.set_ylabels('Reaction time (ms)', size = fontsize)
    p3.set_xticklabels(['',0,.2,.4,.6,.8,1,''])
    
    # Plot rt against bias2 model certainty
    # Take out RT < 100 ms  
    sns.set_context('poster')
    subj_df = plot_df.query('rt > 100 & id < "%s"' %plot_ids[3])       
    p4 = sns.lmplot(x ='bias2_certainty', y = 'rt', hue = 'id', col = 'id', size = 6, data = subj_df)   
    p4.set_xlabels("Model Confidence", size = fontsize)
    p4.set_ylabels('Reaction time (ms)', size = fontsize)
    p4.set_xticklabels(['',0,.2,.4,.6,.8,1,''])
    
    p5 = sns.lmplot(x ='bias2_certainty', y = 'rt', hue = 'id', ci = None, legend = False, size = figdims[1], data = plot_df.query('rt>100'))  
    plt.xlim(-.1,1.1)
    p5.set_xlabels("Model Confidence", size = fontsize)
    p5.set_ylabels('Reaction time (ms)', size = fontsize)
    
    
    # plot bias2 parameters
    params_df = pd.DataFrame()
    params_df['id'] = [x[1:3] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df['learner'] = [x[0:3] in plot_ids for x in bias2_fit_dict if ('_fullRun' in x)] 
    params_df['r1'] = [bias2_fit_dict[x]['r1'] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df['r2'] = [bias2_fit_dict[x]['r2'] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df['eps'] = [bias2_fit_dict[x]['TS_eps'] for x in bias2_fit_dict if ('_fullRun' in x)]
    params_df = pd.melt(params_df, id_vars = ['id','learner'], value_vars = ['eps','r1','r2'], var_name = 'param', value_name = 'val')

    p6 = plt.figure(figsize = figdims)
    ax = plt.subplot(111)
    box_palette = sns.color_palette(['m','c'], desat = 1)
    sns.boxplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, palette = box_palette)
    plt.xlabel("Parameter", size = fontsize)
    plt.ylabel('Value', size = fontsize)
    plt.title('Bias-2 Model Parameter Fits', size = fontsize+4, y = 1.05)
    plt.xticks([0,1,2], ('$\epsilon$','$r_1$','$r_2$'), size = fontsize)
    ax.legend(ax.get_legend_handles_labels()[0],['Learners','Non-learners'], loc = 'upper left')
    
    

    #look at models
    p7 = plt.figure(figsize = figdims)
    plt.hold(True)
    for c in log_posteriors.columns[:-1]:
        sns.kdeplot(summary[c])
    
    p8 = plt.figure(figsize = figdims)
    sns.heatmap(model_subj_compare)
    p9 = plt.figure(figsize = figdims)
    sns.heatmap(model_subj_compare.filter(regex='bias|eoptimal|ignore|subj_ts').corr())
    
    #********** Behavioral Plots **************************
    # look at RT
    p10 = plt.figure(figsize = figdims)
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
    sns.kdeplot(plot_df.query('subj_ts == 0')['rt'], color = 'm', lw = 5, label = 'TS1')
    sns.kdeplot(plot_df.query('subj_ts == 1')['rt'], color = 'c', lw = 5, label = 'TS2')
    plot_df.query('subj_ts == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    plot_df.query('subj_ts == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('Reaction Time (ms)', size = fontsize)
    pylab.legend(loc='upper right',prop={'size':20})
    plt.xlim(xmin=0)
    
    
    #***********************************
    # learner nonlearner behavioral plots
    #***********************************
    plot_df = pd.concat([df,df_fail])
    df.groupby(['last_TS','context']).subj_ts.mean().reset_index()    
    
    p11 = plt.figure(figsize = figdims)
    p11.subplots_adjust(hspace=.3, wspace = .3)
    
    plt.subplot2grid((2,2),(0,0))
    sns.plt.plot(delays,learner_params, 'b-o', label = 'Learners', markersize = 10)
    sns.plt.plot(delays,nonlearner_params, 'r-o', label = 'Non-Learners', markersize = 10)
    plt.xlabel('Context Lag', size = fontsize)
    plt.ylabel('Beta Weights', size = fontsize)
    pylab.legend(loc='best',prop={'size':20})
    plt.tick_params(labelsize=15)
    
    plt.subplot2grid((2,2),(1,0), colspan = 1)
    sns.plt.scatter(range(len(group_means)),group_means, c = [['r','b'][i] for i in label])
    plt.ylabel('Accuracy', size = fontsize)
    plt.xlabel('Subject Index', size = fontsize)
    plt.xlim([-5,50])
    plt.tick_params(labelsize=15)
    
    plt.subplot2grid((2,2),(1,1), colspan = 1)
    sns.plt.plot(k,k_error, '-o')
    plt.ylabel('SSE', size = fontsize)
    plt.xlabel('Number of Clusters (k)', size = fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot2grid((2,2),(0,1))
    for window in [(0,850)]:
        window_df = plot_df.query('trial_count >= %s and trials_since_switch < 27 and trial_count < %s' % (window[0], window[1]))
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
    plt.tick_params(labelsize=15)
    plt.ylabel('Percent Correct', size = fontsize)
    plt.xlabel('Trials Since Objective TS Switch', size = fontsize)

    if save == True:
        p1.savefig('../Plots/TS2%_vs_context.png', format = 'png', dpi = 300)
        p2.savefig('../Plots/Individual_subject_fits.png',format = 'png', dpi = 300)
        p3.savefig('../Plots/rt_vs_posterior_3subj.png', format = 'png', dpi = 300)
        p4.savefig('../Plots/rt_vs_confidence_3subj.png', format = 'png', dpi = 300)
        p5.savefig('../Plots/rt_vs_confidence.png', format = 'png', dpi = 300)
        p6.savefig('../Plots/bias2_param_value.png', format = 'png', dpi = 300)
        p7.savefig('../Plots/model_comparison.png', format = 'png', dpi = 300)
        p10.savefig('../Plots/RTs.png', format = 'png')
        p11.savefig('../Plots/Learner_vs_NonLearner.png', format = 'png', dpi = 300)
        plt.close('all')
        
        