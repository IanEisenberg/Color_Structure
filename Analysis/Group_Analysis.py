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

fullInfo = True

gtrain_df = pd.DataFrame()
gtest_df = pd.DataFrame()
gtaskinfo = []
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


    #*********************************************
    # Set up observers
    #*********************************************
    
    #This observer know the exact statistics of the task, always chooses correctly
    #given that it chooses the correct task-set, and perfectly learns from feedback.
    #This means that it sets the prior probability for each ts to the transition probabilities
    #of the correct task-set on each trial (which a subject 'could' do due to the
    #deterministic feedback). Basically, after receiving FB, the ideal observer
    #knows exactly what task it is in and should act accordingly.
    
    observer_prior = [.5,.5]
    observer_choices = []
    for i,trial in train_dfa.iterrows():
        c = trial.context
        ts = trial.ts
        conf= calc_posterior(c,observer_prior,ts_dis)    
        obs_choice = np.argmax(conf)
        observer_choices.append(obs_choice)
        observer_prior = np.round([.9*(1-ts)+.1*ts,.9*ts+.1*(1-ts)],2)
        
    train_dfa['opt_observer_choices'] = observer_choices
    train_dfa['opt_observer_switch'] = abs((train_dfa.opt_observer_choices).diff())
    train_dfa['conform_opt_observer'] = np.equal(train_dfa.subj_ts, observer_choices)
    
    #Optimal observer for train, without feedback     
    no_fb_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = 1, recursive_prob = train_recursive_p)
    observer_choices = []
    posteriors = []
    for i,trial in train_dfa.iterrows():
        c = trial.context
        posteriors.append(no_fb_observer.calc_posterior(c)[1])
    posteriors = np.array(posteriors)
    train_dfa['no_fb_observer_posterior'] = posteriors
    train_dfa['opt_observer_choices'] = (posteriors>.5).astype(int)
    train_dfa['no_fb_observer_switch'] = (train_dfa.no_fb_observer_posterior>.5).diff()
    train_dfa['conform_no_fb_observer'] = np.equal(train_dfa.subj_ts, posteriors>.5)
    
    #Optimal observer for test        
    optimal_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = 1, recursive_prob = train_recursive_p)
    observer_choices = []
    posteriors = []
    for i,trial in test_dfa.iterrows():
        c = trial.context
        posteriors.append(optimal_observer.calc_posterior(c)[1])
    posteriors = np.array(posteriors)
    
    ##Fix the INT
    test_dfa['opt_observer_posterior'] = posteriors
    test_dfa['opt_observer_choices'] = (posteriors>.5).astype(int)
    test_dfa['opt_observer_switch'] = (test_dfa.opt_observer_posterior>.5).diff()
    test_dfa['conform_opt_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
    
    test_dfa['midline_observer_choices'] = (test_dfa.context_sign == 1).astype('int')
    test_dfa['midline_observer_switch'] = abs((test_dfa.midline_observer_choices).diff())
    test_dfa['conform_midline_observer'] = np.equal(test_dfa.subj_ts, test_dfa.midline_observer_choices)

    train_dfa['id'] = subj_name
    test_dfa['id'] = subj_name
    gtrain_df = pd.concat([gtrain_df,train_dfa])
    gtest_df = pd.concat([gtest_df,test_dfa])   
    gtaskinfo.append(taskinfo)
    
gtaskinfo = pd.DataFrame(gtaskinfo)





#*********************************************
# Switch Analysis
#*********************************************
#Count the number of times there was a switch to each TS for each context value
switch_counts = odict()
switch_counts['midline_observer'] = gtest_df.query('midline_observer_switch == True').groupby(['midline_observer_choices','context']).trial_count.count().unstack(level = 0)
switch_counts['subject'] = gtest_df.query('subj_switch == True').groupby(['subj_ts','context']).trial_count.count().unstack(level = 0)
switch_counts['opt_observer'] = gtest_df.query('opt_observer_switch == True').groupby(['opt_observer_choices','context']).trial_count.count().unstack(level = 0)

#normalize switch counts by the midline rule. The midline rule represents
#the  number of switches someone would make if they switched task-sets
#every time the stimuli's position crossed the midline to that position
norm_switch_counts = odict()
for key in switch_counts:
    empty_df = pd.DataFrame(index = np.unique(gtest_df.context), columns = [0,1])
    empty_df.index.name = 'context'
    empty_df.loc[switch_counts[key].index] = switch_counts[key]
    switch_counts[key] = empty_df
    norm_switch_counts[key] = switch_counts[key].div(switch_counts['midline_observer'],axis = 0)

behav_sum['switch_counts'] = switch_counts['subject']
behav_sum['ts2_side'] = ts2_side
behav_sum['norm_switch_counts'] = norm_switch_counts['subject']

gtest_df.query('opt_observer_switch == True').groupby('context').mean().opt_observer_posterior



#*********************************************
# Plotting
#*********************************************

gtest_df = gtest_df.query('id != "Pilot021"')
ids = np.unique(gtest_df.id)

#Plot task-set count by context value
plt.hold(True) 
plt.plot(gtest_df.groupby('context').subj_ts.mean(), lw = 3, color = 'c', label = 'Subject')
plt.plot(gtest_df.groupby('context').opt_observer_choices.mean(), lw = 3, color = 'c', ls = '--', label = 'optimal observer')
plt.plot(gtest_df.groupby('context').midline_observer_choices.mean(), lw = 3, color = 'c', ls = ':', label = 'midline rule')
plt.xticks(list(range(12)),np.round(list(sub.index),2))
plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
plt.xlabel('Stimulus Vertical Position')
plt.ylabel('Task-set 2 %')
pylab.legend(loc='upper right',prop={'size':20})
for subj in ids:
    subj_df = gtest_df.query('id == "%s"' %subj)
    plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2, color = 'k', alpha = .1)

#plot distribution of switches, by task-set
plt.hold(True) 
sub = switch_counts['subject']
plt.plot(sub[0], lw = 4, color = 'm', label = 'switch to ts 1')
plt.plot(sub[1], lw = 4, color = 'c', label = 'switch to ts 2')
sub = switch_counts['opt_observer']
plt.plot(sub[0], lw = 4, color = 'm', ls = '--', label = 'optimal observer')
plt.plot(sub[1], lw = 4, color = 'c', ls = '--')
sub = switch_counts['midline_observer']
plt.plot(sub[0], lw = 4, color = 'm', ls = '-.', label = 'midline rule')
plt.plot(sub[1], lw = 4, color = 'c', ls = '-.')
plt.xticks(list(range(12)),np.round(list(sub.index),2))
plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
plt.xlabel('Stimulus Vertical Position')
plt.ylabel('Counts')
pylab.legend(loc='upper right',prop={'size':20})
for subj in ids:
    subj_df = gtest_df.query('id == "%s"' %subj)
    subj_switch_counts = odict()
    subj_switch_counts['midline_observer'] = subj_df.query('midline_observer_switch == True').groupby(['midline_observer_choices','context']).trial_count.count().unstack(level = 0)
    subj_switch_counts['subject'] = subj_df.query('subj_switch == True').groupby(['subj_ts','context']).trial_count.count().unstack(level = 0)
    subj_switch_counts['opt_observer'] = subj_df.query('opt_observer_switch == True').groupby(['opt_observer_choices','context']).trial_count.count().unstack(level = 0)
    
    #normalize switch counts by the midline rule. The midline rule represents
    #the  number of switches someone would make if they switched task-sets
    #every time the stimuli's position crossed the midline to that position
    subj_norm_switch_counts = odict()
    for key in subj_switch_counts:
        empty_df = pd.DataFrame(index = np.unique(subj_df.context), columns = [0,1])
        empty_df.index.name = 'context'
        empty_df.loc[switch_counts[key].index] = subj_switch_counts[key]
        subj_switch_counts[key] = empty_df*len(ids)
        subj_norm_switch_counts[key] = subj_switch_counts[key].div(subj_switch_counts['midline_observer'],axis = 0)
    sub = subj_switch_counts['subject']
    plt.plot(sub[0], lw = 3, color = 'm', alpha = .15)
    plt.plot(sub[1], lw = 3, color = 'c', alpha = .15)
#    sub = switch_counts['opt_observer']
#    plt.plot(sub[0], lw = 3, color = 'm', ls = '--', alpha = .15)
#    plt.plot(sub[1], lw = 3, color = 'c', ls = '--', alpha = .15)


    
#As above, using normalized measure
plt.hold(True) 
sub = norm_switch_counts['subject']
plt.plot(sub[0], lw = 4, color = 'm', label = 'switch to ts 1')
plt.plot(sub[1], lw = 4, color = 'c', label = 'switch to ts 2')
sub = norm_switch_counts['opt_observer']
plt.plot(sub[0], lw = 4, color = 'm', ls = '--', label = 'optimal observer')
plt.plot(sub[1], lw = 4, color = 'c', ls = '--')
sub = norm_switch_counts['midline_observer']
plt.plot(sub[0], lw = 4, color = 'm', ls = '-.', label = 'midline rule')
plt.plot(sub[1], lw = 4, color = 'c', ls = '-.')
plt.xticks(list(range(12)),np.round(list(sub.index),2))
plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
plt.xlabel('Stimulus Vertical Position')
plt.ylabel('Normalized Counts Compared to Midline Rule')
pylab.legend(loc='best',prop={'size':20})
for subj in ids:
    subj_df = gtest_df.query('id == "%s"' %subj)
    subj_switch_counts = odict()
    subj_switch_counts['midline_observer'] = subj_df.query('midline_observer_switch == True').groupby(['midline_observer_choices','context']).trial_count.count().unstack(level = 0)
    subj_switch_counts['subject'] = subj_df.query('subj_switch == True').groupby(['subj_ts','context']).trial_count.count().unstack(level = 0)
    subj_switch_counts['opt_observer'] = subj_df.query('opt_observer_switch == True').groupby(['opt_observer_choices','context']).trial_count.count().unstack(level = 0)
    
    #normalize switch counts by the midline rule. The midline rule represents
    #the  number of switches someone would make if they switched task-sets
    #every time the stimuli's position crossed the midline to that position
    subj_norm_switch_counts = odict()
    for key in subj_switch_counts:
        empty_df = pd.DataFrame(index = np.unique(subj_df.context), columns = [0,1])
        empty_df.index.name = 'context'
        empty_df.loc[switch_counts[key].index] = subj_switch_counts[key]
        subj_switch_counts[key] = empty_df*len(ids)
        subj_norm_switch_counts[key] = subj_switch_counts[key].div(subj_switch_counts['midline_observer'],axis = 0)
    sub = subj_norm_switch_counts['subject']
    plt.plot(sub[0], lw = 3, color = 'm', alpha = .15)
    plt.plot(sub[1], lw = 3, color = 'c', alpha = .15)
#    sub = switch_counts['opt_observer']
#    plt.plot(sub[0], lw = 3, color = 'm', ls = '--', alpha = .15)
#    plt.plot(sub[1], lw = 3, color = 'c', ls = '--', alpha = .15)
    
    
#look at RT
plt.subplot(4,1,1)
gtest_df.rt.hist(bins = 25)
plt.ylabel('Count across subject')

plt.subplot(4,1,2)
plt.hold(True)
gtest_df.query('subj_switch == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'stay')
gtest_df.query('subj_switch == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'switch')
gtest_df.query('subj_switch == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
gtest_df.query('subj_switch == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
plt.xlabel('RT')
pylab.legend(loc='upper right',prop={'size':20})

plt.subplot(4,1,3)
plt.hold(True)
gtest_df.query('subj_switch == 0 and rep_resp == 1')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'repeat response')
gtest_df.query('subj_switch == 0 and rep_resp == 0')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'change response (within task-set)')
gtest_df.query('subj_switch == 0 and rep_resp == 1')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
gtest_df.query('subj_switch == 0 and rep_resp == 0')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
plt.xlabel('RT')
plt.ylabel('Normed Count')
pylab.legend(loc='upper right',prop={'size':20})

plt.subplot(4,1,4)
plt.hold(True)
gtest_df.query('subj_ts == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'ts1')
gtest_df.query('subj_ts == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'ts2')
gtest_df.query('subj_ts == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
gtest_df.query('subj_ts == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
plt.xlabel('RT')
pylab.legend(loc='upper right',prop={'size':20})