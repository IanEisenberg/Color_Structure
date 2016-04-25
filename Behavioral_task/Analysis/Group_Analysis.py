"""
Created on Mon Apr 27 11:16:08 2015

@author: Ian
"""

import numpy as np
from scipy.stats import norm
from Load_Data import load_data
from helper_classes import BiasPredModel, SwitchModel
from helper_functions import fit_bias2_model, fit_bias1_model, fit_static_model, \
    fit_switch_model, fit_midline_model, calc_posterior, gen_TS_posteriors
import pickle, glob, re
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
plot = False
save = True

# *********************************************
# Load Data
# ********************************************
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

if save is False:
    gtest_learn_df = pd.DataFrame.from_csv('Analysis_Output/gtest_learn_df.csv')
    gtest_df = pd.DataFrame.from_csv('Analysis_Output/gtest_df.csv')
    gtest_learn_df.id = gtest_learn_df.id.astype('str')
    gtest_df.id = gtest_df.id.astype('str')
    gtest_learn_df.id = gtest_learn_df.id.apply(lambda x: x.zfill(3))
    gtest_df.id = gtest_df.id.apply(lambda x: x.zfill(3))
else:
    group_behavior = {}
    gtrain_df = pd.DataFrame()
    gtest_df = pd.DataFrame()
    gtaskinfo = []

    train_files = sorted(glob.glob(data_dir + '/Mega/IanE_RawData/Prob_Context_Task/RawData/*Context_20*yaml'))
    test_files = sorted(glob.glob(data_dir + '/Mega/IanE_RawData/Prob_Context_Task/RawData/*Context_test*yaml'))

    count = 0

    for train_file, test_file in zip(train_files, test_files):
        subj_name = re.match(r'.*/RawData.(\w*)_Prob*', test_file).group(1)
        print(subj_name)
        if subj_name in ['034']:
            pass
        count += 1
        train_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', train_file).group(1)
        test_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', test_file).group(1)
        try:
            train_dict = pickle.load(open('../Data/' + train_name + '.p', 'rb'))
            taskinfo, train_dfa = [train_dict.get(k) for k in ['taskinfo', 'dfa']]

        except FileNotFoundError:
            train_taskinfo, train_dfa = load_data(train_file, train_name, mode='train')
            train_dict = {'taskinfo': train_taskinfo, 'dfa': train_dfa}
            pickle.dump(train_dict, open('../Data/' + train_name + '.p','wb'))

        try:
            test_dict = pickle.load(open('../Data/' + test_name + '.p','rb'))
            taskinfo, test_dfa = [test_dict.get(k) for k in ['taskinfo','dfa']]
        except FileNotFoundError:
            taskinfo, test_dfa = load_data(test_file, test_name, mode='test')
            test_dict = {'taskinfo': taskinfo, 'dfa': test_dfa}
            pickle.dump(test_dict, open('../Data/' + test_name + '.p','wb'))

    # *********************************************
    # Preliminary Setup
    # *********************************************

        recursive_p = taskinfo['recursive_p']
        states = taskinfo['states']
        state_dis = [norm(states[0]['c_mean'], states[0]['c_sd']), norm(states[1]['c_mean'], states[1]['c_sd']) ]
        ts_order = [states[0]['ts'],states[1]['ts']]
        ts_dis = [state_dis[i] for i in ts_order]
        ts2_side = np.sign(ts_dis[1].mean())
        taskinfo['ts2_side'] = ts2_side
        # To ensure TS2 is always associated with the 'top' of the screen, or positive
        # context values, flip the context values if this isn't the case.
        # This ensures that TS1 is always the shape task-set and, for analysis purposes,
        # always associated with the bottom of the screen
        train_dfa['true_context'] = train_dfa['context']
        test_dfa['true_context'] = test_dfa['context']

        if ts2_side == -1:
            train_dfa['context'] = train_dfa['context']* -1
            test_dfa['context'] = test_dfa['context']* -1
            ts_dis = ts_dis [::-1]

        # What was the mean contextual value for each taskset during this train run?
        train_ts_means = list(train_dfa.groupby('ts').agg(np.mean).context)
        # Same for standard deviation
        train_ts_std = list(train_dfa.groupby('ts').agg(np.std).context)
        train_ts_dis = [norm(m, s) for m, s in zip(train_ts_means, train_ts_std)]
        # And do the same for recursive_p
        train_recursive_p = 1 - train_dfa.switch.mean()
        #how often did the response not match either of the stim's features
        action_eps = 1-np.mean([test_dfa['response'][i] in test_dfa['stim'][i] for i in test_dfa.index])

        # decompose contexts
        test_dfa['abs_context'] = abs(test_dfa.context)
        train_dfa['abs_context'] = abs(train_dfa.context)
        train_dfa['context_sign'] = np.sign(train_dfa.context)
        test_dfa['context_sign'] = np.sign(test_dfa.context)
        # Create vector of context differences
        test_dfa['context_diff'] = test_dfa['context'].diff()

        # transform rt
        train_dfa['log_rt'] = np.log(train_dfa.rt)
        test_dfa['log_rt'] = np.log(test_dfa.rt)
        
        # *********************************************
        # Model fitting
        # *********************************************
        df_midpoint = round(len(test_dfa)/2)
        for model_type in ['TS', 'action']:
            print(model_type)
            if subj_name + '_' + model_type + '_first' not in bias2_fit_dict.keys():
                bias2_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_bias2_model(train_ts_dis, test_dfa, action_eps = action_eps, model_type = model_type)
                bias2_fit_dict[subj_name + '_' + model_type + '_first']  = fit_bias2_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], action_eps = action_eps, model_type = model_type)
                bias2_fit_dict[subj_name + '_' + model_type + '_second']  = fit_bias2_model(train_ts_dis, test_dfa.iloc[df_midpoint:], action_eps = action_eps, model_type = model_type)
            if subj_name + '_' + model_type + '_first' not in bias1_fit_dict.keys():    
                bias1_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_bias1_model(train_ts_dis, test_dfa, action_eps = action_eps, model_type = model_type)
                bias1_fit_dict[subj_name + '_' + model_type + '_first']  = fit_bias1_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], action_eps = action_eps, model_type = model_type)
                bias1_fit_dict[subj_name + '_' + model_type + '_second']  = fit_bias1_model(train_ts_dis, test_dfa.iloc[df_midpoint:], action_eps = action_eps, model_type = model_type)
            if subj_name + '_' + model_type + '_first' not in eoptimal_fit_dict.keys():                
                eoptimal_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_static_model(train_ts_dis, test_dfa, train_recursive_p, action_eps = action_eps, model_type = model_type)
                eoptimal_fit_dict[subj_name + '_' + model_type + '_first']  = fit_static_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], train_recursive_p, action_eps = action_eps, model_type = model_type)
                eoptimal_fit_dict[subj_name + '_' + model_type + '_second']  = fit_static_model(train_ts_dis, test_dfa.iloc[df_midpoint:], train_recursive_p, action_eps = action_eps, model_type = model_type)
            if subj_name + '_' + model_type + '_first' not in ignore_fit_dict.keys():                
                ignore_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_static_model(train_ts_dis, test_dfa, .5, action_eps = action_eps, model_type = model_type)                
                ignore_fit_dict[subj_name + '_' + model_type + '_first']  = fit_static_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], .5, action_eps = action_eps, model_type = model_type)
                ignore_fit_dict[subj_name + '_' + model_type + '_second']  = fit_static_model(train_ts_dis, test_dfa.iloc[df_midpoint:], .5, action_eps = action_eps, model_type = model_type)
            if subj_name + '_first' not in midline_fit_dict.keys():               
                midline_fit_dict[subj_name + '_fullRun'] = fit_midline_model(test_dfa)                
                midline_fit_dict[subj_name + '_first'] = fit_midline_model(test_dfa.iloc[0:df_midpoint])
                midline_fit_dict[subj_name + '_second'] = fit_midline_model(test_dfa.iloc[df_midpoint:])
            if subj_name + '_first' not in switch_fit_dict.keys():    
                switch_fit_dict[subj_name + '_fullRun'] = fit_switch_model(test_dfa)                
                switch_fit_dict[subj_name + '_first'] = fit_switch_model(test_dfa.iloc[0:df_midpoint])
                switch_fit_dict[subj_name + '_second'] = fit_switch_model(test_dfa.iloc[df_midpoint:])
    
        
        # *********************************************
        # Set up observers
        # *********************************************
        
        # **************TRAIN*********************
        # This observer know the exact statistics of the task, always chooses correctly
        # given that it chooses the correct task-set, and perfectly learns from feedback.
        # This means that it sets the prior probability for each ts to the transition probabilities
        # of the correct task-set on each trial (which a subject 'could' do due to the
        # deterministic feedback). Basically, after receiving FB, the ideal observer
        # knows exactly what task it is in and should act accordingly.
        observer_prior = [.5,.5]
        observer_choices = []
        for i,trial in train_dfa.iterrows():
            c = trial.context
            ts = trial.ts
            conf= calc_posterior(c,observer_prior,ts_dis)    
            obs_choice = np.argmax(conf)
            observer_choices.append(obs_choice)
            observer_prior = np.round([.9*(1-ts)+.1*ts,.9*ts+.1*(1-ts)],2)
            
        train_dfa['opt_choices'] = observer_choices
        train_dfa['opt_switch'] = abs((train_dfa.opt_choices).diff())
        train_dfa['conform_opt'] = np.equal(train_dfa.subj_ts, observer_choices)
        
        # Optimal observer for train, without feedback     
        no_fb = BiasPredModel(train_ts_dis, [.5,.5], r1 = train_recursive_p, r2 = train_recursive_p, TS_eps = 0, action_eps = action_eps)
        observer_choices = []
        posteriors = []
        for i,trial in train_dfa.iterrows():
            c = trial.context
            posteriors.append(no_fb.calc_posterior(c)[1])
        posteriors = np.array(posteriors)
        train_dfa['no_fb_posterior'] = posteriors
        train_dfa['opt_choices'] = (posteriors>.5).astype(int)
        train_dfa['no_fb_switch'] = (train_dfa.no_fb_posterior>.5).diff()
        train_dfa['conform_no_fb'] = np.equal(train_dfa.subj_ts, posteriors>.5)
        
        
        # **************TEST*********************
        model_type = 'TS'
        # Bias2 observer for test    
        params = bias2_fit_dict[subj_name + '_' + model_type + '_fullRun']
        bias2 = BiasPredModel(train_ts_dis, [.5,.5],**params)
        params = bias1_fit_dict[subj_name + '_' + model_type + '_fullRun']
        bias1 = BiasPredModel(train_ts_dis, [.5,.5], **params)
        params = eoptimal_fit_dict[subj_name + '_' + model_type + '_fullRun']
        eoptimal = BiasPredModel(train_ts_dis, [.5,.5], **params)
        params = ignore_fit_dict[subj_name + '_' + model_type + '_fullRun']
        ignore = BiasPredModel(train_ts_dis, [.5,.5], **params)
        
        # Fit observer for test        
        gen_TS_posteriors([bias2, bias1, eoptimal, ignore], test_dfa, ['bias2', 'bias1', 'eoptimal', 'ignore'])        
        
        # midline observer for test  
        eps = midline_fit_dict[subj_name + '_fullRun']['eps']
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = max(0,np.sign(trial.context))
            posteriors.append(abs(c - eps))
        posteriors = np.array(posteriors)
    
        test_dfa['midline_posterior'] = posteriors
    
        # Switch observer for test  
        params = switch_fit_dict[subj_name + '_fullRun']      
        switch = SwitchModel(**params)
        posteriors = []
        for i,trial in test_dfa.iterrows():
            if i == 0:
                 last_choice = -1 
            else:
                last_choice = test_dfa.subj_ts[i-1]
            trial_choice = trial.subj_ts
            conf = switch.calc_TS_prob(last_choice)
            posteriors.append(conf[trial_choice])           
        posteriors = np.array(posteriors)
        
        test_dfa['switch_posterior'] = posteriors

    
        for model in ['bias2','bias1','eoptimal', 'ignore', 'midline', 'switch']:
            test_dfa[model + '_choices'] = (test_dfa[model + '_posterior']>.5).astype(int)
            test_dfa[model + '_certainty'] = (abs(test_dfa[model + '_posterior']-.5))/.5
        
        #test_dfa['bias2_choices'] = (posteriors>.5).astype(int)
        #test_dfa['bias2_switch'] = (test_dfa.bias2_posterior>.5).diff()
    
        train_dfa['id'] = subj_name
        test_dfa['id'] = subj_name
        gtrain_df = pd.concat([gtrain_df,train_dfa])
        gtest_df = pd.concat([gtest_df,test_dfa])   
        gtaskinfo.append(taskinfo)
    
     
    gtaskinfo = pd.DataFrame(gtaskinfo)
    
    # Exclude subjects where stim_confom is below some threshold 
    select_ids = gtest_df.groupby('id').mean().stim_conform>.75
    select_ids = select_ids[select_ids]
    select_rows = [i in select_ids for i in gtrain_df.id]
    gtrain_df = gtrain_df[select_rows]
    select_rows = [i in select_ids for i in gtest_df.id]
    gtest_df = gtest_df[select_rows]
    ids = select_ids.index
    
    # separate learner group
    select_ids = gtest_df.groupby('id').mean().correct > .55
    select_ids = select_ids[select_ids]
    select_rows = [i in select_ids for i in gtrain_df.id]
    gtrain_learn_df = gtrain_df[select_rows]
    select_rows = [i in select_ids for i in gtest_df.id]
    gtest_learn_df = gtest_df[select_rows]
    learn_ids = select_ids.index


    pickle.dump(bias2_fit_dict,open('Analysis_Output/bias2_parameter_fits.p','wb'), protocol=2)
    pickle.dump(bias1_fit_dict,open('Analysis_Output/bias1_parameter_fits.p','wb'), protocol=2)
    pickle.dump(eoptimal_fit_dict,open('Analysis_Output/eoptimal_parameter_fits.p','wb'), protocol=2)
    pickle.dump(ignore_fit_dict,open('Analysis_Output/ignore_parameter_fits.p','wb'), protocol=2)
    pickle.dump(midline_fit_dict,open('Analysis_Output/midline_parameter_fits.p','wb'), protocol=2)
    pickle.dump(switch_fit_dict,open('Analysis_Output/switch_parameter_fits.p','wb'), protocol=2)
    gtest_learn_df.to_csv('Analysis_Output/gtest_learn_df.csv')
    gtest_df.to_csv('Analysis_Output/gtest_df.csv')
    gtrain_learn_df.to_csv('Analysis_Output/gtrain_learn_df.csv')
    gtrain_df.to_csv('Analysis_Output/gtrain_df.csv')
    gtaskinfo.to_csv = ('Analysis_Output_gtaskinfo.csv')

# *********************************************
# Calculate how well the bias-2 model does
# ********************************************* 
r1 = pd.concat([gtest_learn_df['id'], gtest_learn_df['subj_ts']==gtest_learn_df['bias1_choices']],1)
np.mean(r1.groupby('id').mean())

np.mean([bias2_fit_dict[w + '_fullRun']['r2'] for w in plot_ids] + [bias2_fit_dict[w + '_fullRun']['r1'] for w in plot_ids])
# *********************************************
# Model Comparison
# ********************************************* 
compare_df = gtest_learn_df
compare_df_subset= compare_df[['subj_ts','bias2_posterior','bias1_posterior','eoptimal_posterior','ignore_posterior','midline_posterior','switch_posterior']]
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

import statsmodels.formula.api as smf
import statsmodels.api as sm

switch_sums = []
trials_since_switch = 0
for i,row in gtest_learn_df.iterrows():
    if row['switch'] == 1 or pd.isnull(row['switch']):
        trials_since_switch = 0
    else:
        trials_since_switch += 1
    switch_sums.append(trials_since_switch)
gtest_learn_df['trials_since_switch'] = switch_sums

df = gtest_learn_df
res = smf.ols(formula='correct ~ trials_since_switch + rt', data=df).fit()

res = sm.GLM(df['correct'], df[['trials_since_switch']], family = sm.families.Binomial()).fit()
res = sm.OLS(df['bias2_certainty'], df[['trials_since_switch']]).fit()

print(res.summary())

plot_df = {}
for i in np.unique(df['id']):
    temp_df = df.query('id == "%s"' % i)
    plot_df[i] = [temp_df.query('trials_since_switch == %s' % i)['correct'].mean() for i in np.unique(temp_df['trials_since_switch'])]
plot_df = pd.DataFrame.from_dict(plot_df, orient='index').transpose()  

plot_df = plot_df.unstack().reset_index(name = 'percent_correct')
plot_df.rename(columns={'level_0': 'id', 'level_1': 'trials_since_switch'}, inplace=True)
sns.lmplot(x = 'trials_since_switch', y = 'percent_correct', data = plot_df)
smf.ols(formula = 'percent_correct ~ trials_since_switch', data = plot_df).fit().summary()

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

    
    p11 = plt.figure(figsize = figdims)
    plt.hold(True)
    for c in log_posteriors.columns[:-1]:
        sns.kdeplot(summary[c])
        
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
        
