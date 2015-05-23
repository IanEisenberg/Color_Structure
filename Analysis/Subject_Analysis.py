# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:12:43 2015

@author: Ian
"""


import numpy as np
from scipy.stats import norm
import lmfit
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from Load_Data import load_data
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import OrderedDict as odict
from helper_classes import PredModel, BiasPredModel
from helper_functions import *
from ggplot import *
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

save = False
plot = False
fitting = True

#*********************************************
# Load Data
#*********************************************

train_files = glob.glob('../RawData/*Context_20*yaml')
test_files = glob.glob('../RawData/*Context_test*yaml')

subj_i = 15
train_file = train_files[subj_i]
test_file = test_files[subj_i]

test_name = test_file[11:-5]
train_name = train_file[11:-5]
subj_name = re.match(r'(\w*)_Prob*', test_name).group(1)

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


behav_sum = odict()

#*********************************************
# Model fitting
#*********************************************

if fitting == True:
    #*************************************
    #Model Functions
    #*************************************
    
    def bias_fitfunc(rp, contexts, choices, tsb):
        model = BiasPredModel(train_ts_dis, init_prior, ts_bias = tsb, recursive_prob = rp)
        model_likelihoods = []
        for i,c in enumerate(contexts):
            trial_choice = choices[i]
            conf = model.calc_posterior(c)
            model_likelihoods.append(conf[trial_choice])
        return np.array(model_likelihoods)
        
    def bias_errfunc(params,contexts,choices):
        rp = params['rp'].value
        tsb = params['tsb'].value
        #minimize:
        return abs(np.log(bias_fitfunc(rp,contexts,choices,tsb))) #log posterior for each choice
        #return abs(np.sum(np.log(bias_fitfunc(rp,contexts,choices,tsb)))) #single value
        
    init_prior = [.5,.5]
    
    #Fit bias model
    #attempt to simplify:
    fit_params = lmfit.Parameters()
    fit_params.add('rp', value = .5, min = 0, max = 1)
    fit_params.add('tsb', value = .5, min = 0)
    out = lmfit.minimize(bias_errfunc,fit_params, method = 'lbfgsb', kws= {'contexts':list(test_dfa.context), 'choices':list(test_dfa.subj_ts)})
    fit_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = out.values['tsb'], recursive_prob = out.values['rp'])
    lmfit.report_fit(out)
    
    #Fit observer for test        
    observer_choices = []
    posteriors = []
    for i,trial in test_dfa.iterrows():
        c = trial.context
        posteriors.append(fit_observer.calc_posterior(c)[1])
    posteriors = np.array(posteriors)

    test_dfa['fit_observer_posterior'] = posteriors
    test_dfa['fit_observer_choices'] = (posteriors>.5).astype(int)
    test_dfa['fit_observer_switch'] = (test_dfa.fit_observer_posterior>.5).diff()
    test_dfa['conform_fit_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
    test_dfa['fit_certainty'] = (abs(test_dfa.fit_observer_posterior-.5))/.5
#*********************************************
# Set up caricature observers
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
train_dfa['opt_observer_switch'] = (train_dfa.opt_observer_choices).diff()
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

test_dfa['opt_observer_posterior'] = posteriors
test_dfa['opt_observer_choices'] = (posteriors>.5).astype(int)
test_dfa['opt_observer_switch'] = (test_dfa.opt_observer_posterior>.5).diff()
test_dfa['conform_opt_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
test_dfa['opt_certainty'] = (abs(test_dfa.opt_observer_posterior-.5))/.5

#Ignore observer for test        
ignore_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = 1, recursive_prob = .5)
observer_choices = []
posteriors = []
for i,trial in test_dfa.iterrows():
    c = trial.context
    posteriors.append(ignore_observer.calc_posterior(c)[1])
posteriors = np.array(posteriors)
test_dfa['ignore_observer_posterior'] = posteriors
test_dfa['ignore_observer_choices'] = (posteriors>.5).astype(int)
test_dfa['ignore_observer_switch'] = (test_dfa.ignore_observer_posterior>.5).diff()
test_dfa['conform_ignore_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)

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
# Switch training accuracy
#*********************************************

behav_sum['train_switch_acc'] = train_dfa.groupby('subj_switch').conform_opt_observer.mean()[1]

#*********************************************
# Contributors to task-set choice
#*********************************************
sub = sm.add_constant(test_dfa[['context_sign','abs_context','context','subj_ts','rt']])
sub['last_ts'] = sub.subj_ts.shift(1)
predictors = sub.drop(['subj_ts'],axis = 1)
result = smf.logit(formula = 'subj_ts ~ context + last_ts', data = sub, missing = 'drop').fit()


#*********************************************
# Models
#*********************************************

model_subj_compare = test_dfa[['subj_ts','fit_observer_posterior', 'opt_observer_posterior', 'ignore_observer_posterior']].corr()

fit_log_posterior = np.sum(np.log([abs(test_dfa.subj_ts.loc[i] - (1-test_dfa.fit_observer_posterior.loc[i])) for i in test_dfa.index]))
midline_rule_log_posterior = np.sum(np.log([abs(test_dfa.subj_ts.loc[i] - (1-abs(test_dfa.ignore_observer_choices.loc[i]-.2))) for i in test_dfa.index]))

init_prior = [.5,.5]
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
# Plotting
#*********************************************


if plot == True:
    
    contexts = np.unique(test_dfa.context)
    figdims = (16,12)
    #Plot conformity to a number of measures (models, experimental ts, ignore)
    plt.figure(figsize = figdims)
    plt.subplot(2,1,1)
    plt.hold(True)
    plt.plot(pd.ewma(np.equal(train_dfa.context_sign==1,train_dfa.subj_ts), span = 50), lw = 3, label = 'ignore rule')
    plt.plot(pd.ewma(train_dfa.conform_opt_observer,span = 50), lw = 3, label = 'optimal')  
    plt.plot(pd.ewma(np.equal(train_dfa.ts,train_dfa.subj_ts),span = 50), lw = 3, label = 'experiment TS')  
    plt.plot(pd.ewma(train_dfa.conform_no_fb_observer,span = 50), lw = 3, label = 'optimal no FB') 
    plt.axhline(.5, color = 'k', lw = 3, ls = '--')
    plt.ylabel('EWMA (span = 50) conformity')
    plt.title('Training')
    pylab.legend(loc='lower right',prop={'size':20})
    
    
    plt.subplot(2,1,2)
    plt.hold(True)
    plt.title('Testing')
    plt.plot(pd.ewma(test_dfa.conform_ignore_observer, span = 50), lw = 3, label = 'ignore rule')
    plt.plot(pd.ewma(test_dfa.conform_opt_observer,span = 50), lw = 3, label = 'optimal')  
    plt.plot(pd.ewma(np.equal(test_dfa.ts,test_dfa.subj_ts),span = 50), lw = 3, label = 'experiment TS')  
    plt.xlabel('Trial')
    plt.axhline(.5, color = 'k', lw = 3, ls = '--')
    pylab.legend(loc='lower right',prop={'size':20})
    
    p1=plt.figure(figsize = figdims)
    #Plot task-set count by context value
    plt.hold(True) 
    plt.plot(test_dfa.groupby('context').subj_ts.mean(), lw = 3, color = 'c', label = 'Subject')
    plt.plot(test_dfa.groupby('context').opt_observer_choices.mean(), lw = 3, color = 'c', ls = '--', label = 'optimal observer')
    plt.plot(test_dfa.groupby('context').ignore_observer_choices.mean(), lw = 3, color = 'c', ls = ':', label = 'ignore rule')
    plt.xticks(list(range(12)),contexts)
    plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
    plt.xlabel('Stimulus Vertical Position')
    plt.ylabel('Task-set 2 %')
    pylab.legend(loc='best',prop={'size':20})
    
    
    #plot distribution of switches, by task-set
    p2=plt.figure(figsize = figdims)
    plt.subplot(2,1,1)
    plt.hold(True) 
    sub = switch_counts['subject']
    plt.plot(sub[0], lw = 3, color = 'm', label = 'switch to ts 1')
    plt.plot(sub[1], lw = 3, color = 'c', label = 'switch to ts 2')
    sub = switch_counts['opt_observer']
    plt.plot(sub[0], lw = 3, color = 'm', ls = '--', label = 'optimal observer')
    plt.plot(sub[1], lw = 3, color = 'c', ls = '--')
    sub = switch_counts['ignore_observer']
    plt.plot(sub[0], lw = 3, color = 'm', ls = '-.', label = 'ignore rule')
    plt.plot(sub[1], lw = 3, color = 'c', ls = '-.')
    plt.xticks(list(range(12)),contexts)
    plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
    plt.xlabel('Stimulus Vertical Position')
    plt.ylabel('Counts')
    pylab.legend(loc='upper right',prop={'size':20})
    
    #As above, using normalized measure
    plt.subplot(2,1,2)
    plt.hold(True) 
    sub = norm_switch_counts['subject']
    plt.plot(sub[0], lw = 3, color = 'm', label = 'switch to ts 1')
    plt.plot(sub[1], lw = 3, color = 'c', label = 'switch to ts 2')
    sub = norm_switch_counts['opt_observer']
    plt.plot(sub[0], lw = 3, color = 'm', ls = '--', label = 'optimal observer')
    plt.plot(sub[1], lw = 3, color = 'c', ls = '--')
    sub = norm_switch_counts['ignore_observer']
    plt.plot(sub[0], lw = 3, color = 'm', ls = '-.', label = 'ignore rule')
    plt.plot(sub[1], lw = 3, color = 'c', ls = '-.')
    plt.xticks(list(range(12)),contexts)
    plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
    plt.xlabel('Stimulus Vertical Position')
    plt.ylabel('Normalized Counts Compared to ignore Rule')
    pylab.legend(loc='best',prop={'size':20})
    
    #look at RT
    p3=plt.figure(figsize = figdims)
    plt.subplot(4,1,1)
    plt.plot(test_dfa.rt*1000,'ro')
    plt.title('RT over experiment', size = 24)
    plt.xlabel('trial')
    plt.ylabel('RT in ms')
    
    plt.subplot(4,1,2)
    plt.hold(True)
    test_dfa.query('subj_switch == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'stay')
    test_dfa.query('subj_switch == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'switch')
    test_dfa.query('subj_switch == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    test_dfa.query('subj_switch == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('Normed Count')
    pylab.legend(loc='upper right',prop={'size':20})
    
    plt.subplot(4,1,3)
    plt.hold(True)
    test_dfa.query('subj_switch == 0 and rep_resp == 1')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'repeat response')
    test_dfa.query('subj_switch == 0 and rep_resp == 0')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'change response (within task-set)')
    test_dfa.query('subj_switch == 0 and rep_resp == 1')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    test_dfa.query('subj_switch == 0 and rep_resp == 0')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('Normed Count')
    pylab.legend(loc='upper right',prop={'size':20})
    
    plt.subplot(4,1,4)
    plt.hold(True)
    test_dfa.query('subj_ts == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'ts1')
    test_dfa.query('subj_ts == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'ts2')
    test_dfa.query('subj_ts == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    test_dfa.query('subj_ts == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('Normed Count')
    pylab.legend(loc='upper right',prop={'size':20})
    

    #RT for switch vs stay for different trial-by-trial context diff
    test_dfa.groupby(['subj_switch','context_diff']).mean().rt.unstack(level = 0).plot(kind='bar', color = ['c','m'], figsize = figdims)     
    
    #Plot rt against optimal model certainty
    ggplot(test_dfa, aes('opt_certainty', 'rt')) + geom_point() + geom_smooth(method = 'lm')

    ggplot(test_dfa, aes('fit_certainty', 'rt')) + geom_point() + geom_smooth(method = 'lm')
    
    #Plot run
    plotting_dict = {'optimal': ['optimal', 'b','optimal'],
                    'single': ['single', 'c','TS(t-1)'],
                     'ignore': ['ignore', 'r','base rate neglect']}
    sub = dfa_modeled[50:200]
    plot_run(sub,plotting_dict, exclude = ['single'])
    if save == True:
        plt.savefig('../Plots/' +  subj_name + '_summary_plot.png', dpi = 300, bbox_inches='tight')





