# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:57:14 2015

@author: Ian
"""

import random as r
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
from ggplot import * 
from helper_classes import PredModel,DataGenerator, BiasPredModel
from datetime import datetime
from collections import OrderedDict as odict

#***********************
#Analysis
#***********************

#Make random subject with some noise values following one of the models and try
#to predict
std_track ={}
for std in [.37]:
    startTime = datetime.now()
    init_prior = [.5, .5]
    exp_len = 600
    model_choice = ['ignore','single','optimal']
    subj_choices = ['norm','noisy','prob_match','noisy_prob_match','softmax']
    
    
    #Set up dataframes to record choices
    posterior_keys = ['subj_i','subj_mode','choice_mode','ts','context','subj','ignore','single','optimal']
    choices_keys = ['subj_i','subj_mode','choice_mode','ts','context','subj','ignore','single','optimal']
    perform_keys = ['subj_i','subj_mode','choice_mode','ts','context','subj','ignore','single','optimal']
    likelihoods_keys = ['subj_i','subj_mode','choice_mode','ts','context','ignore','single','optimal','rand','ts0','ts1']
    group_posteriors, group_choices, group_performs,group_likelihoods = [],[],[],[]
    
    u = .3
    ts_dis = [norm(-u,std),norm(u,std)]    
    for subj_i in range(1):
        print(subj_i)
        
        
        #Setup new models and set an even prior
        models = [ \
            PredModel(ts_dis, init_prior, mode = "ignore"),\
            PredModel(ts_dis, init_prior, mode = "single"),\
            PredModel(ts_dis, init_prior, mode = "optimal"), \
            BalancedPredModel(ts_dis,init_prior, balance = 1)]
        model_prior = [1.0/len(models)]*6
        
        #Generate data, choose a random subject 'mode' and make the subject
        data_gen = DataGenerator(ts_dis,.9)
        trials = [data_gen.gen_data() for _ in range(exp_len)]
        mode = 'optimal' #r.choice(model_choice)
        choice_mode = 'prob_match'
        subj_model = PredModel(ts_dis, init_prior, mode = mode)  
        
        
        for trial in trials:
            trial_num = trial['trial_count']
            c = round(trial['context'],1)
            ts = trial['ts']
            trial_posterior = [subj_i,mode,choice_mode,ts,c,subj_model.calc_posterior(c)[0]]
            trial_choice = [subj_i,mode,choice_mode,ts,c,subj_model.choose(mode = choice_mode, random_prob = .2)]
    
            model_posteriors= []
            model_choices=[]
            trial_model_likelihoods = [subj_i,mode,choice_mode,ts,c]
            trial_perform = [subj_i,mode,choice_mode,ts,c]
            for i,model in enumerate(models):
                conf = model.calc_posterior(c)
                model_posteriors += [conf[0]]
                model_choices += [model.choose()]
                trial_model_likelihoods += [np.log(conf[trial_choice[5]])]
            #add on 'straw model' predictions.
            trial_model_likelihoods += list(np.log([.5,[.9,.1][trial_choice[5]], [.1,.9][trial_choice[5]]]))
            #record
            trial_posterior += model_posteriors            
            trial_choice += model_choices
            trial_perform += list(np.equal(ts,trial_choice[5:])*1)
            trial_posterior
            group_posteriors.append(odict(zip(perform_keys,trial_posterior)))
            group_performs.append(odict(zip(perform_keys,trial_perform)))
            group_choices.append(odict(zip(perform_keys,trial_choice)))
            group_likelihoods.append(odict(zip(likelihoods_keys,trial_model_likelihoods)))
    
    group_posteriors = pd.DataFrame(group_posteriors, columns = posterior_keys)       
    group_performs = pd.DataFrame(group_performs, columns = perform_keys)       
    group_choices = pd.DataFrame(group_choices, columns = choices_keys)       
    group_likelihoods = pd.DataFrame(group_likelihoods, columns = likelihoods_keys)       
    
            
    print(datetime.now()-startTime)
          
            
    
    
    group_posteriors=group_posteriors.convert_objects(convert_numeric = True)
    group_choices=group_choices.convert_objects(convert_numeric = True)
    group_performs=group_performs.convert_objects(convert_numeric = True)
    group_likelihoods=group_likelihoods.convert_objects(convert_numeric = True)
    std_track[std] = {'group_posteriors':group_posteriors,
                        'group_choices': group_choices,
                        'group_likelihoods': group_likelihoods,
                        'group_performs': group_performs}
                        
                        
sd = .5
std_track[sd]['group_performs'].groupby('context').mean()


#******************************************
#Test acc difference between optimal and ignore models
#******************************************

sim_acc = []
for sim in range(20):
    a = DataGenerator(ts_dis,.9)
    trials = [a.gen_data() for _ in range(700)]
    
    optimal_observer = BiasPredModel(ts_dis, [.5,.5], bias = 0, recursive_prob = recursive_p)
    ignore_observer = BiasPredModel(ts_dis, [.5,.5], bias = 1, recursive_prob = recursive_p)
    
    observer_choices = []
    ignore_choices = []
    tss = []
    for i,trial in enumerate(trials):
        tss.append(trial['ts'])
        c = trial['context']
        conf = optimal_observer.calc_posterior(c)
        obs_choice = np.argmax(conf)
        observer_choices.append(obs_choice)
        
        ignore_conf = ignore_observer.calc_posterior(c)
        ignore_choice = np.argmax(ignore_conf)
        ignore_choices.append(ignore_choice)
        
    np.sum(np.equal(tss,observer_choices))/len(tss)
    np.sum(np.equal(tss,ignore_choices))/len(tss)
    sim_acc.append((np.sum(np.equal(tss,observer_choices))/len(tss),
                    np.sum(np.equal(tss,ignore_choices))/len(tss)))
                    
optimal,ignore = [[z[i] for z in sim_acc] for i in (0,1)]
scipy.stats.ttest_ind(optimal,ignore,equal_var = False)