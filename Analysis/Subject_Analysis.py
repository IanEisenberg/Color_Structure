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
plt.rc('figure', figsize = (16,12))

save = False
plot = False


#*********************************************
# Load Data
#*********************************************
fullInfo = True

if fullInfo:
   train_files = glob.glob('../RawData/*FullInfo_20*yaml')
   test_files = glob.glob('../RawData/*FullInfo_noFB*yaml') 
else:
    train_files = glob.glob('../RawData/*Context_20*yaml')
    test_files = glob.glob('../RawData/*Context_noFB*yaml')

subj_i = 3
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

behav_sum = odict()


#*********************************************
# Set up pobservers
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
        
    dfa['opt_observer_choices'] = observer_choices
    dfa['opt_observer_switch'] = abs(dfa.opt_observer_choices.diff())
    dfa['conform_opt_observer'] = np.equal(train_dfa.subj_ts, observer_choices)

#Optimal observer for test        
no_fb_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = 1, recursive_prob = train_recursive_p)
observer_choices = []
for i,trial in train_dfa.iterrows():
    c = trial.context
    conf = no_fb_observer.calc_posterior(c)
    obs_choice = np.argmax(conf)
    observer_choices.append(obs_choice)
train_dfa['no_fb_observer_choices'] = observer_choices
train_dfa['no_fb_observer_switch'] = abs(train_dfa.no_fb_observer_choices.diff())
train_dfa['conform_no_fb_observer'] = np.equal(train_dfa.subj_ts, observer_choices)


#Optimal observer for test        
optimal_observer = BiasPredModel(train_ts_dis, [.5,.5], ts_bias = 1, recursive_prob = train_recursive_p)
observer_choices = []
for i,trial in test_dfa.iterrows():
    c = trial.context
    conf = optimal_observer.calc_posterior(c)
    obs_choice = np.argmax(conf)
    observer_choices.append(obs_choice)
test_dfa['opt_observer_choices'] = observer_choices
test_dfa['opt_observer_switch'] = abs(test_dfa.opt_observer_choices.diff())
test_dfa['conform_opt_observer'] = np.equal(test_dfa.subj_ts, observer_choices)

test_dfa['midline_observer_choices'] = (test_dfa.context_sign == ts2_side).astype('int')
test_dfa['midline_observer_switch'] = abs(test_dfa.midline_observer_choices.diff())
test_dfa['conform_midline_observer'] = np.equal(test_dfa.subj_ts, test_dfa.midline_observer_choices)

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
switch_contexts = odict()
switch_contexts['subject'] = test_dfa.query('subj_switch == True').groupby(['subj_ts','context']).trial_count.count().unstack(level = 0)
switch_contexts['opt_observer'] = test_dfa.query('opt_observer_switch == True').groupby(['opt_observer_choices','context']).trial_count.count().unstack(level = 0)
switch_contexts['midline_observer'] = test_dfa.query('midline_observer_switch == True').groupby(['midline_observer_choices','context']).trial_count.count().unstack(level = 0)





#*********************************************
# linear fit of RT based on absolute context
#*********************************************

result = sm.GLS(np.log(test_dfa.rt),sm.add_constant(test_dfa.abs_context)).fit()
behav_sum['context->rt'] = result.params[1] * int(result.pvalues[1]<.05)


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

def bias_fit_temp(t, dfa, tsb, rp,):
    model = BiasPredModel(train_ts_dis, init_prior, ts_bias = tsb, recursive_prob = rp)
    model_choices = []
    for i,trial in dfa.iterrows():
        c = trial.context
        model.calc_posterior(c)
        model_choices.append(model.choose(mode = 'softmax', inv_temp = t))
    dfa['model_choices'] = model_choices
    return np.sum(np.square(dfa.groupby('context').model_choices.mean()-test_dfa.groupby('context').subj_ts.mean()))
    
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
    model = EstimatePredModel(init_prior,mean = m, std = s, recursive_prob = rp)
    model_choices = []
    for i,trial in dfa.iterrows():
        c = trial.context
        model.calc_posterior(c)
        model_choices.append(model.choose(mode = 'softmax', inv_temp = t))
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
params.add('t', value = 10)
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
params.add('t', value = 5)
modelfit = model.fit(test_dfa.groupby('context').subj_ts.mean(), params, dfa = test_dfa, m=m, s=s, rp=erp)
etemp = modelfit.values['t']

behav_sum['softmax_temps'] = {'bias_temp':btemp, 'estimate_temp':etemp}

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
    
    #Plot conformity to a number of measures (models, experimental ts, midline)
    plt.subplot(3,1,1)
    plt.hold(True)
    plt.plot(pd.ewma(np.equal(train_dfa.context_sign==ts2_side,train_dfa.subj_ts), span = 50), lw = 3, label = 'midline rule')
    plt.plot(pd.ewma(train_dfa.conform_opt_observer,span = 50), lw = 3, label = 'optimal')  
    plt.plot(pd.ewma(train_dfa.correct,span = 50), lw = 3, label = 'experiment TS')  
    plt.plot(pd.ewma(train_dfa.conform_no_fb_observer,span = 50), lw = 3, label = 'optimal no FB') 
    plt.axhline(.5, color = 'k', lw = 3, ls = '--')
    plt.xlabel('Trial')
    plt.ylabel('EWMA (span = 50) conformity')
    pylab.legend(loc='bottom right',prop={'size':20})
    
    plt.subplot(3,1,2)
    plt.hold(True)
    plt.plot(pd.ewma(test_dfa.conform_midline_observer, span = 50), lw = 3, label = 'midline rule')
    plt.plot(pd.ewma(test_dfa.conform_opt_observer,span = 50), lw = 3, label = 'optimal')  
    plt.plot(pd.ewma(test_dfa.correct,span = 50), lw = 3, label = 'experiment TS')  
    plt.axhline(.5, color = 'k', lw = 3, ls = '--')
    pylab.legend(loc='bottom right',prop={'size':20})
    
    #plot distribution of switches, by task-set
    plt.hold(True)
    sub = switch_contexts['subject']
    plt.plot(sub[0], lw = 3, color = 'm', label = 'switch to ts 1')
    plt.plot(sub[1], lw = 3, color = 'c', label = 'switch to ts 2')
    sub = switch_contexts['opt_observer']
    plt.plot(sub[0], lw = 3, color = 'm', ls = '--', label = 'optimal observer')
    plt.plot(sub[1], lw = 3, color = 'c', ls = '--')
    sub = switch_contexts['midline_observer']
    plt.plot(sub[0], lw = 3, color = 'm', ls = ':', label = 'midline rule')
    plt.plot(sub[1], lw = 3, color = 'c', ls = ':')
    plt.xticks(list(range(12)),np.round(list(sub.index),2))
    plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
    pylab.legend(loc='upper right',prop={'size':20})
    
    
    
    #look at RT
    plt.subplot(4,1,1)
    plt.plot(dfa.rt*1000,'ro')
    plt.title('RT over experiment', size = 24)
    plt.xlabel('trial')
    plt.ylabel('RT in ms')
    
    plt.subplot(4,1,2)
    plt.hold(True)
    dfa.query('subj_switch == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'stay')
    dfa.query('subj_switch == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'switch')
    dfa.query('subj_switch == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    dfa.query('subj_switch == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('count')
    pylab.legend(loc='upper right',prop={'size':20})
    
    plt.subplot(4,1,3)
    plt.hold(True)
    dfa.query('subj_switch == 0 and rep_resp == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'repeat response')
    dfa.query('subj_switch == 0 and rep_resp == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'change response (within task-set)')
    dfa.query('subj_switch == 0 and rep_resp == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    dfa.query('subj_switch == 0 and rep_resp == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('count')
    pylab.legend(loc='upper right',prop={'size':20})
    
    plt.subplot(4,1,4)
    plt.hold(True)
    dfa.query('subj_ts == 0')['rt'].plot(kind='density', color = 'm', lw = 5, label = 'ts1')
    dfa.query('subj_ts == 1')['rt'].plot(kind='density', color = 'c', lw = 5, label = 'ts2')
    dfa.query('subj_ts == 0')['rt'].hist(bins = 25, alpha = .4, color = 'm', normed = True)
    dfa.query('subj_ts == 1')['rt'].hist(bins = 25, alpha = .4, color = 'c', normed = True)
    plt.xlabel('RT')
    plt.ylabel('count')
    pylab.legend(loc='upper right',prop={'size':20})
            
    ggplot(dfa, aes(x='context',y='rt', color = 'subj_ts')) + geom_point(alpha=.4) + stat_summary(size = 6)
    ggplot(dfa, aes(x='context',y='conform_opt_observer', color = 'subj_ts')) + stat_summary()
    

    #Plot run
    plotting_dict = {'optimal': ['optimal', 'b','optimal'],
                    'single': ['single', 'c','TS(t-1)'],
                     'ignore': ['ignore', 'r','base rate neglect']}
    sub = dfa_modeled[50:200]
    plot_run(sub,plotting_dict, exclude = ['single'])
    if save == True:
        plt.savefig('../Plots/' +  subj_name + '_summary_plot.png', dpi = 300, bbox_inches='tight')





