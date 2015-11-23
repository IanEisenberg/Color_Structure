"""
Created on Mon Apr 27 11:16:08 2015

@author: Ian
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from Load_Data import load_data
from helper_classes import BiasPredModel, SwitchModel
from helper_functions import *
import statsmodels.api as sm
import pickle, glob, re, os, lmfit
import seaborn as sns
from collections import OrderedDict as odict
import warnings

# Suppress runtimewarning due to pandas bug
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#*********************************************
# Set up defaults
#*********************************************

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20,
        }
        
axes = {'titleweight' : 'bold'
        }
plt.rc('font', **font)
plt.rc('axes', **axes)

plot = True
save = False


#*********************************************
# Load Data
#*********************************************
home = os.path.expanduser("~")
try:
    bias2_fit_dict = pickle.load(open('Analysis_Output/bias2_parameter_fits.p','rb'))
except:
    bias2_fit_dict = {}
try:
    bias1_fit_dict = pickle.load(open('Analysis_Output/bias1_parameter_fits.p','rb'))
except:
    bias1_fit_dict = {}
try:
    eoptimal_fit_dict = pickle.load(open('Analysis_Output/eoptimal_parameter_fits.p','rb'))
except:
    eoptimal_fit_dict = {}
try:
    ignore_fit_dict = pickle.load(open('Analysis_Output/ignore_parameter_fits.p','rb'))
except:
    ignore_fit_dict = {}
try:
    midline_fit_dict = pickle.load(open('Analysis_Output/midline_parameter_fits.p','rb'))
except:
    midline_fit_dict = {}
try:
    switch_fit_dict = pickle.load(open('Analysis_Output/switch_parameter_fits.p','rb'))
except:
    switch_fit_dict = {}
    
if save == False:
    gtest_learn_df = pd.DataFrame.from_csv('Analysis_Output/gtest_learn_df_crossval.csv')
    gtest_df = pd.DataFrame.from_csv('Analysis_Output/gtest_df_crossval.csv')
    gtest_learn_df.id = gtest_learn_df.id.astype('str')
    gtest_df.id = gtest_df.id.astype('str')
    gtest_learn_df.id = gtest_learn_df.id.apply(lambda x: x.zfill(3))
    gtest_df.id = gtest_df.id.apply(lambda x: x.zfill(3))
else:        
    group_behavior = {}
    gtrain_df = pd.DataFrame()
    gtest_df = pd.DataFrame()
    gtaskinfo = []
    
    train_files = glob.glob(home + '/Mega/IanE_RawData/Prob_Context_Task/RawData/*Context_20*yaml')
    test_files = glob.glob(home + '/Mega/IanE_RawData/Prob_Context_Task/RawData/*Context_test*yaml')
        
    count = 0
    for train_file, test_file in zip(train_files,test_files):
        count += 1
        if count != 1:
            pass #continue
        train_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', train_file).group(1)
        test_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', test_file).group(1)
        subj_name = re.match(r'.*/RawData.(\w*)_Prob*', test_file).group(1)
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
        
        #transform rt
        train_dfa['log_rt'] = np.log(train_dfa.rt)
        test_dfa['log_rt'] = np.log(test_dfa.rt)
        
        #*********************************************
        # Model fitting
        #*********************************************
        df_midpoint = round(len(test_dfa)/2)
        if subj_name + '_first' not in bias2_fit_dict.keys():
            #Fitting Functions
            def errfunc(params,df):
                r1 = params['r1']
                r2 = params['r2']
                eps = params['eps']
                
                init_prior = [.5,.5]
                model = BiasPredModel(train_ts_dis, init_prior, r1 = r1, r2 = r2, eps = eps)
                model_likelihoods = []
                for i in df.index:
                    c = df.context[i]
                    trial_choice = df.subj_ts[i]
                    conf = model.calc_posterior(c)
                    model_likelihoods.append(conf[trial_choice])
                #minimize
                return abs(np.sum(np.log(np.array(model_likelihoods)))) #single value
            
            #Fit bias model
            fit_params = lmfit.Parameters()
            fit_params.add('r1', value = .5, min = 0, max = 1)
            fit_params.add('r2', value = .5, min = 0, max = 1)
            fit_params.add('eps', value = .1, min = 0, max = 1)
            first_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[0:df_midpoint]})
            bias2_fit_dict[subj_name + '_first'] = first_out.params.valuesdict()
            fit_params = lmfit.Parameters()
            fit_params.add('r1', value = .5, min = 0, max = 1)
            fit_params.add('r2', value = .5, min = 0, max = 1)
            fit_params.add('eps', value = .1, min = 0, max = 1)
            second_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[df_midpoint:]})
            bias2_fit_dict[subj_name + '_second'] = second_out.params.valuesdict()
    
        if subj_name + '_first' not in  bias1_fit_dict.keys():
            #Fitting Functions
            def errfunc(params,df):
                r1 = params['rp']
                r2 = params['rp']
                eps = params['eps']
                
                init_prior = [.5,.5]
                model = BiasPredModel(train_ts_dis, init_prior, r1 = r1, r2 = r2, eps = eps)
                model_likelihoods = []
                for i in df.index:
                    c = df.context[i]
                    trial_choice = df.subj_ts[i]
                    conf = model.calc_posterior(c)
                    model_likelihoods.append(conf[trial_choice])
                #minimize
                return abs(np.sum(np.log(np.array(model_likelihoods)))) #single value
            
            #Fit bias model
            fit_params = lmfit.Parameters()
            fit_params.add('rp', value = .5, min = 0, max = 1)
            fit_params.add('eps', value = .1, min = 0, max = 1)
            first_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[0:df_midpoint]})
            bias1_fit_dict[subj_name + '_first'] = first_out.params.valuesdict()
            fit_params = lmfit.Parameters()
            fit_params.add('rp', value = .5, min = 0, max = 1)
            fit_params.add('eps', value = .1, min = 0, max = 1)
            second_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[df_midpoint:]})
            bias1_fit_dict[subj_name + '_second'] = second_out.params.valuesdict()
        
        if subj_name + '_first' not in  eoptimal_fit_dict.keys():
            #Fitting Functions
            def errfunc(params,df):
                r1 = train_recursive_p
                r2 = train_recursive_p
                eps = params['eps']
                
                init_prior = [.5,.5]
                model = BiasPredModel(train_ts_dis, init_prior, r1 = r1, r2 = r2, eps = eps)
                model_likelihoods = []
                for i in df.index:
                    c = df.context[i]
                    trial_choice = df.subj_ts[i]
                    conf = model.calc_posterior(c)
                    model_likelihoods.append(conf[trial_choice])
                #minimize
                return abs(np.sum(np.log(np.array(model_likelihoods)))) #single value
            
            #Fit bias model
            fit_params = lmfit.Parameters()
            fit_params.add('eps', value = .1, min = 0, max = 1)
            first_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[0:df_midpoint]})
            eoptimal_fit_dict[subj_name + '_first'] = first_out.params.valuesdict()
            fit_params = lmfit.Parameters()
            fit_params.add('eps', value = .1, min = 0, max = 1)
            second_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[df_midpoint:]})
            eoptimal_fit_dict[subj_name + '_second'] = second_out.params.valuesdict()
        
        #fit ignore rule random probability:
        if subj_name + '_first' not in ignore_fit_dict.keys():
            #Fitting Functions
            def errfunc(params,df):
                eps = params['eps']   
                init_prior = [.5,.5]
                model = BiasPredModel(train_ts_dis, init_prior, r1 = .5, r2 = .5, eps = eps)
                model_likelihoods = []
                for i in df.index:
                    c = df.context[i]
                    trial_choice = df.subj_ts[i]
                    conf = model.calc_posterior(c)
                    model_likelihoods.append(conf[trial_choice])
                #minimize
                return abs(np.sum(np.log(np.array(model_likelihoods)))) #single value
            
            #Fit bias model
            fit_params = lmfit.Parameters()
            fit_params.add('eps', value = .1, min = 0, max = 1)
            first_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[0:df_midpoint]})
            ignore_fit_dict[subj_name + '_first'] = first_out.params.valuesdict()
            fit_params = lmfit.Parameters()
            fit_params.add('eps', value = .1, min = 0, max = 1)
            second_out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[df_midpoint:]})
            ignore_fit_dict[subj_name + '_second'] = second_out.params.valuesdict()
    
        #fit midline rule random probability:
        if subj_name + '_first' not in midline_fit_dict.keys():
            #Fitting Functions
            def midline_errfunc(params,df):
                eps = params['eps'].value
                context_sgn = np.array([max(i,0) for i in df.context_sign])
                choice = df.subj_ts
                #minimize
                return -np.sum(np.log(abs(abs(choice - (1-context_sgn))-eps)))
                
            #Fit bias model
            fit_params = lmfit.Parameters()
            fit_params.add('eps', value = .1, min = 0, max = 1)
            first_out = lmfit.minimize(midline_errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[0:df_midpoint]})
            midline_fit_dict[subj_name + '_first'] = first_out.params.valuesdict()
            fit_params = lmfit.Parameters()
            fit_params.add('eps', value = .1, min = 0, max = 1)
            second_out = lmfit.minimize(midline_errfunc,fit_params, method = 'lbfgsb', kws= {'df':test_dfa.iloc[df_midpoint:]})
            midline_fit_dict[subj_name + '_second'] = second_out.params.valuesdict()
        if subj_name + '_first' not in switch_fit_dict.keys():
            #fit switch rule
            def switch_errfunc(params,df):
                params = params.valuesdict()
                r1 = params['r1']
                r2 = params['r2']   
                model = SwitchModel(r1 = r1, r2 = r2)
                model_likelihoods = []
                model_likelihoods.append(.5)
                for i in df.index[1:]:
                    last_choice = df.subj_ts[i-1]
                    trial_choice = df.subj_ts[i]
                    conf = model.calc_TS_prob(last_choice)
                    model_likelihoods.append(conf[trial_choice])
                    
                #minimize
                return abs(np.sum(np.log(model_likelihoods))) #single value
                
            #Fit switch model
            fit_params = lmfit.Parameters()
            fit_params.add('r1', value = .5, min = 0, max = 1)
            fit_params.add('r2', value = .5, min = 0, max = 1)
            first_out = lmfit.minimize(switch_errfunc,fit_params, method = 'lbfgsb', kws= {'df': test_dfa[0:df_midpoint]})
            switch_fit_dict[subj_name + '_first'] = first_out.params.valuesdict()
            fit_params = lmfit.Parameters()
            fit_params.add('r1', value = .5, min = 0, max = 1)
            fit_params.add('r2', value = .5, min = 0, max = 1)
            second_out = lmfit.minimize(switch_errfunc,fit_params, method = 'lbfgsb', kws= {'df': test_dfa[df_midpoint:]})
            switch_fit_dict[subj_name + '_second'] = second_out.params.valuesdict()
        
        #*********************************************
        # Set up observers
        #*********************************************
            
        #bias2 fit observers
        params = bias2_fit_dict[subj_name + '_first']
        first_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = params['r1'], r2 = params['r2'], eps = params['eps'])
        params = bias2_fit_dict[subj_name + '_second']
        second_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = params['r1'], r2 = params['r2'], eps = params['eps'])
        
        #Fit observer for test        
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = trial.context
            if i<df_midpoint:
                first_fit_observer.calc_posterior(c)
                posteriors.append(second_fit_observer.calc_posterior(c)[1])
            else:
                posteriors.append(first_fit_observer.calc_posterior(c)[1])
                second_fit_observer.calc_posterior(c)
        posteriors = np.array(posteriors)
        
        test_dfa['bias2_observer_posterior'] = posteriors
        test_dfa['bias2_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa['bias2_observer_switch'] = (test_dfa.bias2_observer_posterior>.5).diff()
        test_dfa['conform_bias2_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa['bias2_certainty'] = (abs(test_dfa.bias2_observer_posterior-.5))/.5
        
        #bias1 fit observers
        params = bias1_fit_dict[subj_name + '_first']
        first_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = params['rp'], r2 = params['rp'], eps = params['eps'])
        params = bias1_fit_dict[subj_name + '_second']
        second_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = params['rp'], r2 = params['rp'], eps = params['eps'])
        
        #Fit observer for test        
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = trial.context
            if i<df_midpoint:
                first_fit_observer.calc_posterior(c)
                posteriors.append(second_fit_observer.calc_posterior(c)[1])
            else:
                posteriors.append(first_fit_observer.calc_posterior(c)[1])
                second_fit_observer.calc_posterior(c)
        posteriors = np.array(posteriors)
        
        test_dfa['bias1_observer_posterior'] = posteriors
        test_dfa['bias1_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa['bias1_observer_switch'] = (test_dfa.bias1_observer_posterior>.5).diff()
        test_dfa['conform_bias1_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa['bias1_certainty'] = (abs(test_dfa.bias1_observer_posterior-.5))/.5
        
        #eoptimal fit observers
        params = eoptimal_fit_dict[subj_name + '_first']
        first_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = train_recursive_p, r2 = train_recursive_p, eps = params['eps'])
        params = eoptimal_fit_dict[subj_name + '_second']
        second_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = train_recursive_p, r2 = train_recursive_p, eps = params['eps'])
        
        #Fit observer for test        
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = trial.context
            if i<df_midpoint:
                first_fit_observer.calc_posterior(c)
                posteriors.append(second_fit_observer.calc_posterior(c)[1])
            else:
                posteriors.append(first_fit_observer.calc_posterior(c)[1])
                second_fit_observer.calc_posterior(c)
        posteriors = np.array(posteriors)
        
        test_dfa['eoptimal_observer_posterior'] = posteriors
        test_dfa['eoptimal_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa['eoptimal_observer_switch'] = (test_dfa.eoptimal_observer_posterior>.5).diff()
        test_dfa['conform_eoptimal_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa['eoptimal_certainty'] = (abs(test_dfa.eoptimal_observer_posterior-.5))/.5
        
        #ignorefit observers
        params = ignore_fit_dict[subj_name + '_first']
        first_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = train_recursive_p, r2 = train_recursive_p, eps = params['eps'])
        params = ignore_fit_dict[subj_name + '_second']
        second_fit_observer = BiasPredModel(train_ts_dis, [.5,.5], r1 = train_recursive_p, r2 = train_recursive_p, eps = params['eps'])
        
        #Fit observer for test        
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = trial.context
            if i<df_midpoint:
                first_fit_observer.calc_posterior(c)
                posteriors.append(second_fit_observer.calc_posterior(c)[1])
            else:
                posteriors.append(first_fit_observer.calc_posterior(c)[1])
                second_fit_observer.calc_posterior(c)
        posteriors = np.array(posteriors)
        
        test_dfa['ignore_observer_posterior'] = posteriors
        test_dfa['ignore_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa['ignore_observer_switch'] = (test_dfa.ignore_observer_posterior>.5).diff()
        test_dfa['conform_ignore_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa['ignore_certainty'] = (abs(test_dfa.ignore_observer_posterior-.5))/.5
        
        #midline fit observers
        first_eps = midline_fit_dict[subj_name + '_first']['eps']
        second_eps = midline_fit_dict[subj_name + '_second']['eps']
        
        #Fit observer for test        
        posteriors = []
        for i,trial in test_dfa.iterrows():
            c = max(0,np.sign(trial.context))
            if i<df_midpoint:
                posteriors.append(abs(c - second_eps))
            else:
                posteriors.append(abs(c - first_eps))
        posteriors = np.array(posteriors)
    
        test_dfa['midline_observer_posterior'] = posteriors
        test_dfa['midline_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa['midline_observer_switch'] = (test_dfa.midline_observer_posterior>.5).diff()
        test_dfa['conform_midline_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa['midline_certainty'] = (abs(test_dfa.midline_observer_posterior-.5))/.5    
        
        #Switch observer for test  
        params = switch_fit_dict[subj_name + '_first']      
        first_switch_observer = SwitchModel(r1 = params['r1'], r2 = params['r2'])
        params = switch_fit_dict[subj_name + '_second']      
        second_switch_observer = SwitchModel(r1 = params['r1'], r2 = params['r2'])
        posteriors = []
        for i,trial in test_dfa.iterrows():
            if i == 0:
                 last_choice = -1 
            else:
                last_choice = test_dfa.subj_ts[i-1]
            trial_choice = trial.subj_ts
            if i<df_midpoint:
                conf = second_switch_observer.calc_TS_prob(last_choice)
            else:
                conf = first_switch_observer.calc_TS_prob(last_choice)
            posteriors.append(conf[trial_choice])           
        posteriors = np.array(posteriors)
        
        test_dfa['switch_observer_posterior'] = posteriors
        test_dfa['switch_observer_choices'] = (posteriors>.5).astype(int)
        test_dfa['switch_observer_switch'] = (test_dfa.switch_observer_posterior>.5).diff()
        test_dfa['conform_switch_observer'] = np.equal(test_dfa.subj_ts, posteriors>.5)
        test_dfa['switch_certainty'] = (abs(test_dfa.switch_observer_posterior-.5))/.5 
        
        
        train_dfa['id'] = subj_name
        test_dfa['id'] = subj_name
        gtrain_df = pd.concat([gtrain_df,train_dfa])
        gtest_df = pd.concat([gtest_df,test_dfa])   
        gtaskinfo.append(taskinfo)
    
        
    gtaskinfo = pd.DataFrame(gtaskinfo)
    
    #Exclude subjects where stim_confom is below some threshold 
    select_ids = gtest_df.groupby('id').mean().stim_conform>.75
    select_ids = select_ids[select_ids]
    select_rows = [i in select_ids for i in gtrain_df.id]
    gtrain_df = gtrain_df[select_rows]
    select_rows = [i in select_ids for i in gtest_df.id]
    gtest_df = gtest_df[select_rows]
    ids = select_ids.index
    
    #separate learner group
    select_ids = gtest_df.groupby('id').mean().correct > .55
    select_ids = select_ids[select_ids]
    select_rows = [i in select_ids for i in gtrain_df.id]
    gtrain_learn_df = gtrain_df[select_rows]
    select_rows = [i in select_ids for i in gtest_df.id]
    gtest_learn_df = gtest_df[select_rows]
    learn_ids = select_ids.index
    
    pickle.dump(bias2_fit_dict,open('Analysis_Output/bias2_parameter_fits.p','wb'))
    pickle.dump(bias1_fit_dict,open('Analysis_Output/bias1_parameter_fits.p','wb'))
    pickle.dump(eoptimal_fit_dict,open('Analysis_Output/eoptimal_parameter_fits.p','wb'))
    pickle.dump(ignore_fit_dict,open('Analysis_Output/ignore_parameter_fits.p','wb'))
    pickle.dump(midline_fit_dict,open('Analysis_Output/midline_parameter_fits.p','wb'))
    pickle.dump(switch_fit_dict,open('Analysis_Output/switch_parameter_fits.p','wb'))
    gtest_learn_df.to_csv('Analysis_Output/gtest_learn_df_crossval.csv')
    gtest_df.to_csv('Analysis_Output/gtest_df_crossval.csv')  
    
#*********************************************
# Switch Analysis
#*********************************************
#Count the number of times there was a switch to each TS for each context value
switch_counts = odict()
switch_counts['midline_observer'] = gtest_learn_df.query('midline_observer_switch == True').groupby(['midline_observer_choices','context']).trial_count.count().unstack(level = 0)
switch_counts['subject'] = gtest_learn_df.query('subj_switch == True').groupby(['subj_ts','context']).trial_count.count().unstack(level = 0)
switch_counts['eoptimal_observer'] = gtest_learn_df.query('eoptimal_observer_switch == True').groupby(['eoptimal_observer_choices','context']).trial_count.count().unstack(level = 0)
switch_counts['bias2_observer'] = gtest_learn_df.query('bias2_observer_switch == True').groupby(['bias2_observer_choices','context']).trial_count.count().unstack(level = 0)
switch_counts['bias1_observer'] = gtest_learn_df.query('bias1_observer_switch == True').groupby(['bias1_observer_choices','context']).trial_count.count().unstack(level = 0)
switch_counts['ignore_observer'] = gtest_learn_df.query('ignore_observer_switch == True').groupby(['ignore_observer_choices','context']).trial_count.count().unstack(level = 0)


#normalize switch counts by the ignore rule. The ignore rule represents
#the  number of switches someone would make if they switched task-sets
#every time the stimuli's position crossed the ignore to that position
norm_switch_counts = odict()
for key in switch_counts:
    empty_df = pd.DataFrame(index = np.unique(gtest_df.context), columns = [0,1])
    empty_df.index.name = 'context'
    empty_df.loc[switch_counts[key].index] = switch_counts[key]
    switch_counts[key] = empty_df


#*********************************************
# Model Comparison
#********************************************* 
compare_df = gtest_learn_df
compare_df_subset= compare_df[['subj_ts','bias2_observer_posterior','bias1_observer_posterior','eoptimal_observer_posterior','ignore_observer_posterior','midline_observer_posterior','switch_observer_posterior']]
model_subj_compare = compare_df_subset.corr()

log_posteriors = pd.DataFrame()
for model in compare_df_subset.columns[1:]:
    log_posteriors[model] = np.log(abs(compare_df_subset.subj_ts-(1-compare_df_subset[model])))

compare_df = pd.concat([compare_df[['id','subj_ts','context']], log_posteriors], axis = 1)
compare_df['random_log'] = np.log(.5)

summary = compare_df.groupby('id').sum().drop(['context','subj_ts'],axis = 1)



    
#*********************************************
# Plotting
#*********************************************

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
    plt.plot(plot_df.groupby('context').bias2_observer_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', label = 'bias-2 observer')
    plt.plot(plot_df.groupby('context').bias1_observer_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = 'bias-1 observer')
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
    plt.plot(plot_df.groupby('context').eoptimal_observer_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = '--', label = r'$\epsilon$-optimal observer')
    plt.plot(plot_df.groupby('context').midline_observer_choices.mean(), lw = 4, marker = 'o', markersize = 10, color = 'c', ls = ':', label = 'midline rule')
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
        plt.plot(subj_df.groupby('context').subj_ts.mean(), lw = 2,  alpha = 1)
    plt.gca().set_color_cycle(None)
    subj_df = plot_df.query('id == "%s"' %plot_ids[range_start])
    plt.plot(subj_df.groupby('context').bias2_observer_choices.mean(), lw = 2, ls = '--', label = 'bias-2 observer')
    for subj in plot_ids[range_start+1:range_start+5]:
        subj_df = plot_df.query('id == "%s"' %subj)
        plt.plot(subj_df.groupby('context').bias2_observer_choices.mean(), lw = 2, ls = '--')
    pylab.legend(loc='best',prop={'size':20})
        
    # plot distribution of switches, by task-set
    p3 = plt.figure(figsize = figdims)
    plt.subplot(2,1,1)
    plt.hold(True) 
    sub = switch_counts['subject']
    plt.plot(sub[0], lw = 4, color = 'm', label = 'switch to CTS')
    plt.plot(sub[1], lw = 4, color = 'c', label = 'switch to STS')
    sub = switch_counts['bias2_observer']
    plt.plot(sub[0], lw = 4, color = 'm', ls = '-.', label = 'bias observer')
    plt.plot(sub[1], lw = 4, color = 'c', ls = '-.')
    sub = switch_counts['eoptimal_observer']
    plt.plot(sub[0], lw = 4, color = 'm', ls = '--', label = 'optimal observer')
    plt.plot(sub[1], lw = 4, color = 'c', ls = '--')
    sub = switch_counts['ignore_observer']
    plt.plot(sub[0], lw = 4, color = 'm', ls = ':', label = 'midline rule')
    plt.plot(sub[1], lw = 4, color = 'c', ls = ':')
    plt.xticks(list(range(12)),np.round(list(sub.index),2))
    plt.axvline(5.5, lw = 5, ls = '--', color = 'k')
    plt.xlabel('Stimulus Vertical Position', size = fontsize)
    plt.ylabel('Switch Counts', size = fontsize)
    pylab.legend(loc='upper right',prop={'size':20})
    for subj in plot_ids:
        subj_df = plot_df.query('id == "%s"' %subj)
        subj_switch_counts = odict()
        subj_switch_counts['ignore_observer'] = subj_df.query('ignore_observer_switch == True').groupby(['ignore_observer_choices','context']).trial_count.count().unstack(level = 0)
        subj_switch_counts['subject'] = subj_df.query('subj_switch == True').groupby(['subj_ts','context']).trial_count.count().unstack(level = 0)
        subj_switch_counts['bias2_observer'] = subj_df.query('bias2_observer_switch == True').groupby(['bias2_observer_choices','context']).trial_count.count().unstack(level = 0)
        
        # normalize switch counts by the ignore rule. The ignore rule represents
        # the  number of switches someone would make if they switched task-sets
        # every time the stimuli's position crossed the ignore to that position
        subj_norm_switch_counts = odict()
        for key in subj_switch_counts:
            empty_df = pd.DataFrame(index = np.unique(subj_df.context), columns = [0,1])
            empty_df.index.name = 'context'
            empty_df.loc[switch_counts[key].index] = subj_switch_counts[key]
            subj_switch_counts[key] = empty_df*len(plot_ids)
            subj_norm_switch_counts[key] = subj_switch_counts[key].div(subj_switch_counts['ignore_observer'],axis = 0)
        sub = subj_switch_counts['subject']
        plt.plot(sub[0], lw = 3, color = 'm', alpha = .10)
        plt.plot(sub[1], lw = 3, color = 'c', alpha = .10)
    

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
    p6 = sns.lmplot(x='bias2_observer_posterior',y='rt', hue = 'id', data = subj_df, order = 2, size = 6, col = 'id')
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
    params_df['id'] = [x[1:3] for x in bias2_fit_dict if ('_fullRun' not in x)]
    params_df['learner'] = [x[0:3] in plot_ids for x in bias2_fit_dict if ('_fullRun' not in x)] 
    params_df['run'] = ['first' in x for x in bias2_fit_dict if ('_fullRun' not in x)] 
    params_df['r1'] = [bias2_fit_dict[x]['r1'] for x in bias2_fit_dict if ('_fullRun' not in x)]
    params_df['r2'] = [bias2_fit_dict[x]['r2'] for x in bias2_fit_dict if ('_fullRun' not in x)]
    params_df['eps'] = [bias2_fit_dict[x]['eps'] for x in bias2_fit_dict if ('_fullRun' not in x)]
    params_df = pd.melt(params_df, id_vars = ['id','learner', 'run'], value_vars = ['eps','r1','r2'], var_name = 'param', value_name = 'val')

    p9 = plt.figure(figsize = figdims)
    box_palette = sns.color_palette(['m','c'], desat = 1)
    sns.boxplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, palette = box_palette)
    sns.stripplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, jitter = True, edgecolor = "gray", palette = box_palette)
    plt.xlabel("Parameter", size = fontsize)
    plt.ylabel('Value', size = fontsize)
    plt.title('Bias-2 Model Parameter Fits', size = fontsize+4)
    plt.xticks([0,1,2], ('$\epsilon$','$r_1$','$r_2$'), size = fontsize)
    
    # plot bias1 parameters
    params_df = pd.DataFrame()
    params_df['id'] = [x[1:3] for x in bias1_fit_dict if ('_fullRun' not in x)]
    params_df['learner'] = [x[0:3] in plot_ids for x in bias1_fit_dict if ('_fullRun' not in x)] 
    params_df['r1'] = [bias2_fit_dict[x]['r1'] for x in bias1_fit_dict if ('_fullRun' not in x)]
    params_df['eps'] = [bias2_fit_dict[x]['eps'] for x in bias1_fit_dict if ('_fullRun' not in x)]
    params_df = pd.melt(params_df, id_vars = ['id','learner'], value_vars = ['eps','r1'], var_name = 'param', value_name = 'val')

    p10 = plt.figure(figsize = figdims)
    box_palette = sns.color_palette(['m','c'], desat = 1)
    sns.boxplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, palette = box_palette)
    sns.stripplot(x = 'param', y = 'val', hue = 'learner', hue_order = [1,0], data = params_df, jitter = True, edgecolor = "gray", palette = box_palette)
    plt.xlabel("Parameter", size = fontsize)
    plt.ylabel('Value', size = fontsize)
    plt.title('Bias-1 Model Parameter Fits', size = fontsize+4)
    plt.xticks([0,1,2], ('$\epsilon$','$r_1$'), size = fontsize)

    
    p11 = plt.figure(figsize = figdims)
    plt.hold(True)
    for c in log_posteriors.columns[:-1]:
        sns.kdeplot(summary[c])
            
            
    f = [bias2_fit_dict[x + '_first']['r1'] for x in plot_ids]
    s = [bias2_fit_dict[x + '_second']['r1'] for x in plot_ids]
    plt.plot(s,f,'o')
    plt.plot([0, 1], [0, 1], transform=ax.transAxes)
    
    