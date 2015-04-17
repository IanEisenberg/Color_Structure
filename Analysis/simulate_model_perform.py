# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:04:57 2015

@author: Ian
"""
from scipy.stats import norm
import pandas as pd
from helper_classes import PredModel,DataGenerator


#Modeling
ts_dis = [norm(.3,.37),norm(-.3,.37)]    
data_gen = DataGenerator(ts_dis,.9)
init_prior = [.5,.5]

#***************************************
#Compare no-noise models to models with different noise sources
#***************************************
#setup trials
exp_len = 10000
data_gen = DataGenerator(ts_dis,.9)
trials = [data_gen.gen_data() for _ in range(exp_len)]


#setup models
    #dm = data_noise model
    #mm = mean_noise model
    #sm = std  _noise model
    #nn = no noise model

#mn model
samples = []
n = .1
models = [ \
    PredModel(ts_dis, init_prior, mean_noise = n, mode = "ignore"),\
    PredModel(ts_dis, init_prior, mean_noise = n, mode = "single"),\
    PredModel(ts_dis, init_prior, mean_noise = n, mode = "optimal")]
for trial in trials:
    c = round(trial['context'],1)
    #ensure the same noise for different models
    noise_val = norm().rvs()*n
    sample = [n, c, trial['trial_count'], noise_val]
    for model in models:
        sample.append(model.calc_posterior(c, noise_val = noise_val)[0])
    samples.append(sample)            
mn_samples = pd.DataFrame(samples, columns = ['noise_avg','context','trial','noise','ignore','single','optimal'])
  
#sn model      
samples = []
n = .1
models = [ \
    PredModel(ts_dis, init_prior, std_noise = n, mode = "ignore"),\
    PredModel(ts_dis, init_prior, std_noise = n, mode = "single"),\
    PredModel(ts_dis, init_prior, std_noise = n, mode = "optimal")]

for trial in trials:
    c = round(trial['context'],1)
    #ensure the same noise for different models
    noise_val = norm().rvs()*n
    sample = [n, c, trial['trial_count'], noise_val]
    for model in models:
        sample.append(model.calc_posterior(c, noise_val = noise_val)[0])
    samples.append(sample)            
sn_samples = pd.DataFrame(samples, columns = ['noise_avg','context','trial','noise','ignore','single','optimal'])
   
#dn model        
samples = []
n = .1
models = [ \
    PredModel(ts_dis, init_prior, data_noise = n, mode = "ignore"),\
    PredModel(ts_dis, init_prior, data_noise = n, mode = "single"),\
    PredModel(ts_dis, init_prior, data_noise = n, mode = "optimal")]
for trial in trials:
    c = round(trial['context'],1)
    #ensure the same noise for different models
    noise_val = norm().rvs()*n
    sample = [n, c, trial['trial_count'], noise_val]
    for model in models:
        sample.append(model.calc_posterior(c, noise_val = noise_val)[0])
    samples.append(sample)            
dn_samples = pd.DataFrame(samples, columns = ['noise_avg','context','trial','noise','ignore','single','optimal'])

#nn model - no noise
samples = []
n = 0
models = [ \
    PredModel(ts_dis, init_prior, mode = "ignore"),\
    PredModel(ts_dis, init_prior, mode = "single"),\
    PredModel(ts_dis, init_prior, mode = "optimal")]
for trial in trials:
    c = round(trial['context'],1)
    #ensure the same noise for different models
    noise_val = norm().rvs()*n
    sample = [n, c, trial['trial_count'], noise_val]
    for model in models:
        sample.append(model.calc_posterior(c, noise_val = noise_val)[0])
    samples.append(sample)            
nn_samples = pd.DataFrame(samples, columns = ['noise_avg','context','trial','noise','ignore','single','optimal'])

mn_samples.to_csv('../Data/mn_samples.csv')
sn_samples.to_csv('../Data/sn_samples.csv')
dn_samples.to_csv('../Data/dn_samples.csv')
nn_samples.to_csv('../Data/nn_samples.csv')

  