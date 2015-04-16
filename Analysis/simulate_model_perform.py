# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:04:57 2015

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


#Modeling
ts_dis = [norm(.3,.37),norm(-.3,.37)]    
data_gen = DataGenerator(ts_dis,.9)
init_prior = [.5,.5]
ignore_model = PredModel(ts_dis, init_prior, mode = "ignore")
single_model = PredModel(ts_dis, init_prior, mode = "single")
optimal_model = PredModel(ts_dis, init_prior, mode = "optimal")

#see the effect of different amounts of each noise type. These models don't
#update priors trial by trial, so they just are used to estimate the effect
#of noise types
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

exp_len = 1000
#setup trials
data_gen = DataGenerator(ts_dis,.9)
trials = [data_gen.gen_data() for _ in range(exp_len)]

samples = []
n = .1
#setup models
    #dm = data_noise model
    #mm = mean_noise model
    #sm = std  _noise model
models = [ \
    PredModel(ts_dis, init_prior, data_noise = n, mode = "ignore"),\
    PredModel(ts_dis, init_prior, mean_noise = n, mode = "single"),\
    PredModel(ts_dis, init_prior, std_noise = n, mode = "optimal")]

for trial in trials:
    c = round(trial['context'],1)
    #ensure the same noise for different models
    noise_val = norm().rvs()*n
    for model in models:
        sample = [n, c, model.mode, trial['trial_count'], noise_val, \
                  model.calc_posterior(c, noise_val = noise_val)[0]]
        samples.append(sample)            
samples = pd.DataFrame(samples, columns = ['noise_avg','context','model','trial','noise','posterior_ts0'])
        
        
        tmp = pd.DataFrame(samples).transpose()        
        tmp.columns = ['dm','mm', 'sm']
        tmp['noise'] = n
        tmp['context'] = c
        noise_df = noise_df.append(tmp)



ignore, single, optimal = [],[],[]

data_noise = 0
mean_noise = .2
std_noise = 0
noise_type = 'gaussian'

exp_len = 10000
for _ in range(exp_len):
    ignore.append(ignore_model.calc_posterior(data_noise,mean_noise,std_noise, noise_type))


    
    prior_single = trans_probs[np.argmax(single[-1]),:]
    prior_optimal = np.dot(trans_probs,optimal[-1])
    
    
m_data = trialList.copy()
m_data['ignore'] = [val[0] for val in ignore]
m_data['single'] = [val[0] for val in single]
m_data['optimal'] = [val[0] for val in optimal]


m_data.to_csv('../Data/model_simulation.csv')

#***********************
#Analysis
#***********************

#simulate different kinds of noise
for _ in range(100):
    



#Compute correlation between models for each context value. Non-noisy models
#lead to ~0 correlations due to 'ignore' only taking one value per contextual
#value and 'single' only taking two. 'io' = ignore-optimal, 'is' = ignore-single,
#'so' = single-optimal
con_corrs = pd.DataFrame()
for con in m_data.context.unique():
    sub = m_data.query('context == %s' % con)
    sub = sub.loc[:, sub.dtypes==np.float64]
    sub_corr = sub.corr()
    noisy_corr = sub_corr.iloc[4:7,4:7]
    #get lower triangle
    tril = np.tril(noisy_corr,k = -1)
    #extract non zero values
    flatten = np.extract(tril>0, tril)
    context_series = pd.Series(np.append(con, flatten), ['context', 'n_is', 'n_io', 'n_so'])
    con_corrs = con_corrs.append(context_series, ignore_index = True)
con_corrs = con_corrs.sort('context')

plt.plot(con_corrs.context,con_corrs.n_so)

con_choice_corrs = pd.DataFrame()
for con in m_data.context.unique():
    sub = m_data.query('context == %s' % con)
    sub = sub.loc[:, sub.dtypes==np.float64]
    sub = np.round(sub)
    sub_corr = sub.corr()
    context_series = pd.Series([con, sub_corr.loc['single_noisy','optimal_noisy']], ['context', 'n_so_choice'])
    con_choice_corrs = con_choice_corrs.append(context_series, ignore_index = True)
con_choice_corrs = con_choice_corrs.sort('context')

plt.plot(con_corrs.context,con_corrs.n_so)






#Analysis
ignore_cols = ['c_dis','state','stim','trial_count']
sub = m_data.drop(ignore_cols,1)
pd.scatter_matrix(sub)
pd.scatter_matrix(sub.query('context < .35 and context > -.35'))

sub = m_data
plotting_dict = {'optimal': ['optimal', 'b','optimal'],
                 'single': ['single', 'c','TS(t-1)'],
                  'ignore': ['ignore', 'r','base rate neglect']}
                     
plt.hold(True)
models = []
displacement = 0
#plot model certainty and task-set choices
for arg in plotting_dict.values():
    if arg[2] not in []:
        plt.plot(sub.trial_count,sub[arg[0]]*2,arg[1], label = arg[2], lw = 2)
        plt.plot(sub.trial_count, [int(val>.5)+3+displacement for val in sub[arg[0]]],arg[1]+'o')
        displacement+=.15
        models.append(arg[0])
plt.axhline(1, color = 'y', ls = 'dashed', lw = 2)
plt.axhline(2.5, color = 'k', ls = 'dashed', lw = 3)
#plot subject choices (con_shape = conforming to TS1)
#plot current TS, flipping bit to plot correctly
plt.plot(sub.trial_count,(1-sub.ts)-2, 'go', label = 'operating TS')
plt.plot(sub.trial_count, sub.context/2-1.5,'k', lw = 2, label = 'stimulus height')
plt.plot(sub.trial_count, sub.con_shape+2.85, 'yo', label = 'subject choice')
plt.yticks([-2, -1.5, -1, 0, 1, 2, 3.1, 4.1], [ -1, 0 , 1,'0%', '50%',  '100%', 'TS2 Choice', 'TS1 Choice'])
plt.xlim([min(sub.index)-.5,max(sub.index)])
plt.ylim(-2.5,5)
#subdivide graph
plt.axhline(-.5, color = 'k', ls = 'dashed', lw = 3)
plt.axhline(-1.5, color = 'y', ls = 'dashed', lw = 2)
#axes labels
plt.xlabel('trial number')
plt.ylabel('Predicted P(TS1)')
ax = plt.gca()
ax.yaxis.set_label_coords(-.1, .45)
pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08),
          ncol=3, fancybox=True, shadow=True)