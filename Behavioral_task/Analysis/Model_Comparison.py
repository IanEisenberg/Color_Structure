
import os
import numpy as np
import pickle, glob, re
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm



#*********************************************
# Load Data
#*********************************************
data_dir = os.path.expanduser('~')
bias2_fit_dict = pickle.load(open('Analysis_Output/bias2_parameter_fits.p', 'rb'))
bias1_fit_dict = pickle.load(open('Analysis_Output/bias1_parameter_fits.p', 'rb'))
eoptimal_fit_dict = pickle.load(open('Analysis_Output/eoptimal_parameter_fits.p', 'rb'))
ignore_fit_dict = pickle.load(open('Analysis_Output/ignore_parameter_fits.p', 'rb'))
midline_fit_dict = pickle.load(open('Analysis_Output/midline_parameter_fits.p', 'rb'))
switch_fit_dict = pickle.load(open('Analysis_Output/switch_parameter_fits.p', 'rb'))


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
    train_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', train_file).group(1)
    test_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', test_file).group(1)
    try:
        train_dict = pickle.load(open('../Data/' + train_name + '.p', 'rb'))
        taskinfo, train_dfa = [train_dict.get(k) for k in ['taskinfo', 'dfa']]
    except FileNotFoundError:
        print('%s Train file not found' % subj_name)
    try:
        test_dict = pickle.load(open('../Data/' + test_name + '.p','rb'))
        taskinfo, test_dfa = [test_dict.get(k) for k in ['taskinfo','dfa']]
    except FileNotFoundError:
        print('%s Test file not found' % subj_name)
        
        
        
#*********************************************
# Preliminary Setup
#*********************************************

    
    ts_dis = [norm(taskinfo['states'][s]['c_mean'], taskinfo['states'][s]['c_sd']) for s in [0,1]]
    train_ts_dis,train_recursive_p,action_eps = preproc_data(train_dfa,test_dfa,taskinfo)      
    
    #*********************************************
    # Model fitting
    #*********************************************
       
    
    for bias, fit_observer in [('bias',bias_fit_observer), ('nobias', nobias_fit_observer)]:
        #Fit observer for test        
        observer_choices = []
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = trial.context
            posteriors.append(fit_observer.calc_posterior(c)[1])
        posteriors = np.array(posteriors)

        test_dfa[bias + 'fit_observer_posterior'] = posteriors
        test_dfa[bias +'fit_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa[bias +'fit_observer_switch'] = (test_dfa[bias + 'fit_observer_posterior']>.5).diff()
        test_dfa[bias +'conform_fit_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa[bias +'fit_certainty'] = (abs(test_dfa[bias + 'fit_observer_posterior']-.5))/.5
        
        
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
    
    test_dfa['id'] = subj_name
    gtest_df = pd.concat([gtest_df,test_dfa])   
    gtaskinfo.append(taskinfo)
    
gtaskinfo = pd.DataFrame(gtaskinfo)

#Exclude subjects where stim_confom is below some threshold 
select_ids = gtest_df.groupby('id').mean().stim_conform>.75
select_ids = select_ids[select_ids]
select_rows = [i in select_ids for i in gtest_df.id]
gtest_df = gtest_df[select_rows]
ids = select_ids.index

#separate learner group
select_ids = gtest_df.groupby('id').mean().correct > .55
select_ids = select_ids[select_ids]
select_rows = [i in select_ids for i in gtest_df.id]
gtest_learn_df = gtest_df[select_rows]
learn_ids = select_ids.index   
   
   
   
#*********************************************
# Model Comparison
#********************************************* 
compare_df = gtest_learn_df
  
model_subj_compare = compare_df[['subj_ts','opt_observer_posterior','nobiasfit_observer_posterior', 'biasfit_observer_posterior']].corr()

optfit_log_posterior = np.log(abs(compare_df.subj_ts-(1-compare_df.opt_observer_posterior)))
biasfit_log_posterior = np.log(abs(compare_df.subj_ts-(1-compare_df.biasfit_observer_posterior)))
nobiasfit_log_posterior = np.log(abs(compare_df.subj_ts-(1-compare_df.nobiasfit_observer_posterior)))
midline_rule_log_posterior = np.log(abs(compare_df.subj_ts - (1-abs((compare_df.context_sign==1).astype(int)-.1))))

compare_df = pd.concat([compare_df[['id','subj_ts','context']], optfit_log_posterior, biasfit_log_posterior, nobiasfit_log_posterior, midline_rule_log_posterior], axis = 1)
compare_df.columns = ['id','subj_ts','context','optimal','bias','nobias', 'midline']
compare_df['random_log'] = np.log(.5)

summary = compare_df.groupby('id').sum().drop(['context','subj_ts'],axis = 1)
plt.hold(True)
summary.plot(figsize = (16,12), fontsize = 16)
plt.ylabel('Log Posterior')










    
    
    