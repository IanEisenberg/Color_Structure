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



#load models
nn=pd.DataFrame.from_csv('../Data/nn_samples.csv')
mn=pd.DataFrame.from_csv('../Data/mn_samples.csv')
sn=pd.DataFrame.from_csv('../Data/sn_samples.csv')
dn=pd.DataFrame.from_csv('../Data/dn_samples.csv')




#***********************
#Analysis
#***********************
#compare switches
m_data = mn
tmp = np.round(m_data.iloc[:,4:7])
tmp = pd.concat([tmp,tmp.shift(1)!=tmp],axis = 1)
tmp.columns = ['i_choice','s_choice','o_choice','i_switch','s_switch','o_switch']
m_data = pd.concat([m_data,tmp],axis = 1)

#Compute correlation between models for each context value. Non-noisy models
#lead to ~0 correlations due to 'ignore' only taking one value per contextual
#value and 'single' only taking two. 'io' = ignore-optimal, 'is' = ignore-single,
#'so' = single-optimal
m_data = nn
sample_len = 1000
context_corrs = pd.DataFrame()
for con in abs(m_data.context).unique():
    sample = np.arange(sample_len) + round(r.random()*(len(m_data)-sample_len))-1
    sub = m_data.iloc[sample]
    sub = sub[abs(m_data.context) <= con]
    sub=sub.drop('noise_avg',axis = 1)
    sub = sub.loc[:, sub.dtypes==np.float64]
    choice = np.round(sub)
    like = np.log(sub/(1-sub))
    #calculate corr between three models, and keep lower triangle
    choice_corr = np.tril(choice.corr().iloc[2:5,2:5],k = -1)
    like_corr = np.tril(like.corr().iloc[2:5,2:5],k = -1)
    sub_corr = np.tril(sub.corr().iloc[2:5,2:5],k = -1)
    #extract lower triangle
    choice_flat = np.extract(choice_corr>0, choice_corr)
    like_flat = np.extract(like_corr>0, like_corr)
    sub_flat = np.extract(sub_corr>0, sub_corr)
    tmp = pd.DataFrame([choice_flat,like_flat,sub_flat], columns = ['is','io','so'])
    tmp['context_bound'] = con
    tmp['measure'] = ['thresh','loglike','prob']
    context_corrs = context_corrs.append(tmp)
    
context_corrs = context_corrs.sort('context_bound')  
context_corrs.query('context_bound <= 1').groupby('measure').agg(np.mean)

sub = context_corrs.query('measure == "thresh"')
plt.plot(sub.context_bound,sub.io)

#Make random subject with some noise values following one of the models and try
#to predict
ts_dis = [norm(.3,.37),norm(-.3,.37)]    
init_prior = [.5, .5]
exp_len = 500
m_noise = 0
s_noise = 0
rp_noise = 0
model_choice = ['ignore','single','optimal']
models = [ \
    PredModel(ts_dis, init_prior, mode = "ignore"),\
    PredModel(ts_dis, init_prior, mode = "single"),\
    PredModel(ts_dis, init_prior, mode = "optimal")]

prediction_df = pd.DataFrame(columns = ['subject','n_prediction','rn_prediction'])
random_noisy_df = pd.DataFrame(columns = ['ignore','single','optimal','subj'])
noisy_df = pd.DataFrame(columns = ['ignore','single','optimal','subj'])

for subj in range(100):
    data_gen = DataGenerator(ts_dis,.9)
    trials = [data_gen.gen_data() for _ in range(exp_len)]
    
    conf = pd.DataFrame(columns = ['subj','ignore','single','optimal'])
    choices = pd.DataFrame(columns = ['subj','ignore','single','optimal'])
    noisy_choices = pd.DataFrame(columns = ['subj','ignore','single','optimal'])
    random_noisy_choices = pd.DataFrame(columns = ['subj','ignore','single','optimal'])
    
    subj_model = PredModel(ts_dis, init_prior, mean_noise = m_noise,
                           std_noise = s_noise, rp_noise = rp_noise,
                           mode = r.choice(model_choice))                  
    for trial in trials:
        trial_num = trial['trial_count']
        c = round(trial['context'],1)
        trial_conf,trial_choices,trial_noisy_choices, trial_random_noisy_choices = [],[],[],[]
        trial_conf.append(subj_model.calc_posterior(c)[0])
        trial_choices.append(subj_model.choose())
        trial_noisy_choices.append(subj_model.choose(mode = 'noisy'))
        trial_random_noisy_choices.append(subj_model.choose(mode = "random_noisy"))
        for model in models:
             trial_conf.append(model.calc_posterior(c)[0])
             choice = model.choose()
             trial_choices.append(choice)
             trial_noisy_choices.append(choice)
             trial_random_noisy_choices.append(choice)
        conf.loc[trial_num] = trial_conf
        choices.loc[trial_num] = trial_choices
        noisy_choices.loc[trial_num] = trial_noisy_choices
        random_noisy_choices.loc[trial_num] = trial_random_noisy_choices
    
    noisy_choices_corr = noisy_choices.corr().iloc[1:4,0]
    random_noisy_choices_corr = random_noisy_choices.corr().iloc[1:4,0]
    prediction_df.loc[len(prediction_df)] = [
                    subj_model.mode, 
                    np.argmax(noisy_choices_corr),
                    np.argmax(random_noisy_choices_corr)]
    random_noisy_df.loc[len(prediction_df)]  = random_noisy_choices_corr.append(pd.Series(subj_model.mode,index=['subj']))
    noisy_df.loc[len(prediction_df)]  = noisy_choices_corr.append(pd.Series(subj_model.mode,index=['subj']))
          
          
          
          
#see the effect of different amounts of each noise type. These models don't
#update priors trial by trial, so they just are used to estimate the effect
#of noise types
init_prior = [.5,.5]
noise_levels = np.linspace(0,.5,6)
noise_df = pd.DataFrame()
con = [-.5, 0 , .5]
for c in con:
    for n in noise_levels:
        n = round(n,1)
        c = round(c,1)
        #dm = data_noise model
        #mm = mean_noise model
        #sm = std  _noise model
        dm = PredModel(ts_dis, init_prior, data_noise = n, mode = "ignore")
        mm = PredModel(ts_dis, init_prior, mean_noise = n, mode = "ignore")
        sm = PredModel(ts_dis, init_prior, std_noise = n, mode = "ignore")
        models = [dm,mm,sm]
        samples = [[model.calc_posterior(c)[0] for _ in range(100)] for model in models]
        tmp = pd.DataFrame(samples).transpose()        
        tmp.columns = ['dm','mm', 'sm']
        tmp['noise'] = n
        tmp['context'] = c
        noise_df = noise_df.append(tmp)

#reference for syntax
#noise_df_2 = pd.melt(noise_df,id_vars = ['noise', 'context'],value_vars = ['dm','mm','sm'], var_name = 'model', value_name = 'conf')
#ggplot(noise_df_2.query('noise != 0 and context == 0'), aes('conf', fill = 'model')) + geom_histogram() + \
#    facet_wrap('noise') 