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
from helper_classes import PredModel,DataGenerator
from datetime import datetime

#***********************
#Analysis
#***********************

#Make random subject with some noise values following one of the models and try
#to predict
startTime = datetime.now()
ts_dis = [norm(.3,.37),norm(-.3,.37)]    
init_prior = [.5, .5]
exp_len = 500
model_choice = ['ignore','single','optimal']
subj_choices = ['norm','noisy','prob_match','noisy_prob_match']
model_distribution_df = pd.DataFrame(columns = ['ignore','single','optimal','rand','ts0','ts1','subj'])
prediction_df = pd.DataFrame(columns = ['subject','choices','noisy_choices','prob_match','noisy_prob_match'])
choices_df = pd.DataFrame(columns = ['ignore','single','optimal','subj','choice_mode'])

for subj in range(25):
    print(subj)

    #Setup new models and set an even prior
    models = [ \
    PredModel(ts_dis, init_prior, mode = "ignore"),\
    PredModel(ts_dis, init_prior, mode = "single"),\
    PredModel(ts_dis, init_prior, mode = "optimal")]
    model_prior = [1.0/len(models)]*6
    
    #Generate data, choose a random subject 'mode' and make the subject
    data_gen = DataGenerator(ts_dis,.9)
    trials = [data_gen.gen_data() for _ in range(exp_len)]
    mode = r.choice(model_choice)
    choice_mode = r.choice(subj_choices)
    subj_model = PredModel(ts_dis, init_prior, mode = mode)  
                
    #Set up dataframes to record choices
    posteriors = pd.DataFrame(columns = ['subj','ignore','single','optimal'])
    choices = pd.DataFrame(columns = ['subj','ignore','single','optimal'])
    
    #first three are the inference models of interest. Second three are 'straw models'
    #which choose randomly, always ts0, or always ts1, respectively
    model_likelihoods = pd.DataFrame(columns = ['ignore','single','optimal','rand','ts0','ts1'])
    
    for trial in trials:
        trial_num = trial['trial_count']
        c = round(trial['context'],1)
        trial_posterior = [subj_model.calc_posterior(c)[0]]
        trial_choice = [subj_model.choose(mode = choice_mode, random_prob = .2)]

        model_posteriors= []
        model_choices=[]
        trial_model_likelihoods = []
        for i,model in enumerate(models):
            conf = model.calc_posterior(c)
            model_posteriors += [conf[0]]
            model_choices += [model.choose()]
            trial_model_likelihoods += [conf[trial_choice[0]]]
        #add on 'straw model' predictions.
        trial_model_likelihoods += [.5,float(trial_choice[0] == 0), float(trial_choice[0] == 1)] 
        model_likelihoods.loc[trial_num] = trial_model_likelihoods
        
        #record
        trial_posterior += model_posteriors            
        trial_choice += model_choices
        posteriors.loc[trial_num] = trial_posterior
        choices.loc[trial_num] = trial_choice
    
    choices_corr = choices.corr().iloc[1:4,0]
    tmp = np.product(model_likelihoods,axis = 0)
    model_posterior = np.round(tmp/sum(tmp),2)
    model_distribution_df.loc[len(prediction_df)] = np.append(model_posterior,[mode,choice_mode])
    prediction_df.loc[len(prediction_df)] = [
                    mode, 
                    np.argmax(choices_corr),
                    np.argmax(noisy_choices_corr),
                    np.argmax(prob_match_corr),
                    np.argmax(noisy_prob_match_corr)]
    choices_df.loc[len(prediction_df)]  = choices_corr.append(pd.Series(mode,index=['subj']))

         
print(datetime.now()-startTime)
          