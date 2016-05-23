"""
Created on Mon Apr 27 11:16:08 2015

@author: Ian
"""
import os
import numpy as np
from Load_Data import load_data
from helper_classes import BiasPredModel, SwitchModel, MemoryModel
from helper_functions import fit_bias2_model, fit_bias1_model, fit_static_model, \
    fit_switch_model, fit_midline_model, fit_memory_model, calc_posterior, gen_bias_TS_posteriors, \
    gen_memory_TS_posteriors, preproc_data
import pickle, glob, re
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import warnings

# Suppress runtimewarning due to pandas bug
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# *********************************************
# Set up defaults
# *********************************************
plot = False
save = True

# *********************************************
# Load Data
# ********************************************
data_dir = os.path.expanduser('~')
try:
    bias2_fit_dict = pickle.load(open('Analysis_Output/bias2_parameter_fits.p', 'rb'))
except:
    bias2_fit_dict = {}
try:
    bias1_fit_dict = pickle.load(open('Analysis_Output/bias1_parameter_fits.p', 'rb'))
except:
    bias1_fit_dict = {}
try:
    eoptimal_fit_dict = pickle.load(open('Analysis_Output/eoptimal_parameter_fits.p', 'rb'))
except:
    eoptimal_fit_dict = {}
try:
    ignore_fit_dict = pickle.load(open('Analysis_Output/ignore_parameter_fits.p', 'rb'))
except:
    ignore_fit_dict = {}
try:
    midline_fit_dict = pickle.load(open('Analysis_Output/midline_parameter_fits.p', 'rb'))
except:
    midline_fit_dict = {}
try:
    switch_fit_dict = pickle.load(open('Analysis_Output/switch_parameter_fits.p', 'rb'))
except:
    switch_fit_dict = {}
try:
    memory_fit_dict = pickle.load(open('Analysis_Output/memory_parameter_fits.p', 'rb'))
except:
    memory_fit_dict = {}
try:
    perseverance_fit_dict = pickle.load(open('Analysis_Output/perseverance_parameter_fits.p', 'rb'))
except:
    perseverance_fit_dict = {}
try:
    permem_fit_dict = pickle.load(open('Analysis_Output/permem_parameter_fits.p', 'rb'))
except:
    permem_fit_dict = {}

if save is False:
    gtrain_learn_df = pd.read_pickle('Analysis_Output/gtrain_learn_df.pkl')
    gtest_learn_df = pd.read_pickle('Analysis_Output/gtest_learn_df.pkl')
    gtest_conform_df = pd.read_pickle('Analysis_Output/gtest_conform_df.pkl')
    gtest_df = pd.read_pickle('Analysis_Output/gtest_df.pkl')
    gtrain_learn_df.id = gtrain_learn_df.id.astype('str').apply(lambda x: x.zfill(3))
    gtest_learn_df.id = gtest_learn_df.id.astype('str').apply(lambda x: x.zfill(3))

else:
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
        count += 1
        train_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', train_file).group(1)
        test_name = re.match(r'.*/RawData.([0-9][0-9][0-9].*).yaml', test_file).group(1)
        try:
            train_dict = pickle.load(open('../Data/' + train_name + '.p', 'rb'))
            taskinfo, train_dfa = [train_dict.get(k) for k in ['taskinfo', 'dfa']]

        except FileNotFoundError:
            train_taskinfo, train_dfa = load_data(train_file, train_name, mode='train')
            train_dict = {'taskinfo': train_taskinfo, 'dfa': train_dfa}
            pickle.dump(train_dict, open('../Data/' + train_name + '.p','wb'))

        try:
            test_dict = pickle.load(open('../Data/' + test_name + '.p','rb'))
            taskinfo, test_dfa = [test_dict.get(k) for k in ['taskinfo','dfa']]
        except FileNotFoundError:
            taskinfo, test_dfa = load_data(test_file, test_name, mode='test')
            test_dict = {'taskinfo': taskinfo, 'dfa': test_dfa}
            pickle.dump(test_dict, open('../Data/' + test_name + '.p','wb'))

    # *********************************************
    # Preliminary Setup
    # *********************************************
        ts_dis = [norm(taskinfo['states'][s]['c_mean'], taskinfo['states'][s]['c_sd']) for s in [0,1]]
        train_ts_dis,train_recursive_p,action_eps = preproc_data(train_dfa,test_dfa,taskinfo)        
        
        # *********************************************
        # Model fitting
        # *********************************************
        df_midpoint = round(len(test_dfa)/2)
        for model_type in ['TS', 'action']:
            print(model_type)
            if subj_name + '_' + model_type + '_first' not in bias2_fit_dict.keys():
                bias2_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_bias2_model(train_ts_dis, test_dfa, action_eps = action_eps, model_type = model_type)
                bias2_fit_dict[subj_name + '_' + model_type + '_first']  = fit_bias2_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], action_eps = action_eps, model_type = model_type)
                bias2_fit_dict[subj_name + '_' + model_type + '_second']  = fit_bias2_model(train_ts_dis, test_dfa.iloc[df_midpoint:], action_eps = action_eps, model_type = model_type)
            if subj_name + '_' + model_type + '_first' not in bias1_fit_dict.keys():    
                bias1_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_bias1_model(train_ts_dis, test_dfa, action_eps = action_eps, model_type = model_type)
                bias1_fit_dict[subj_name + '_' + model_type + '_first']  = fit_bias1_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], action_eps = action_eps, model_type = model_type)
                bias1_fit_dict[subj_name + '_' + model_type + '_second']  = fit_bias1_model(train_ts_dis, test_dfa.iloc[df_midpoint:], action_eps = action_eps, model_type = model_type)
            if subj_name + '_' + model_type + '_first' not in eoptimal_fit_dict.keys():                
                eoptimal_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_static_model(train_ts_dis, test_dfa, train_recursive_p, action_eps = action_eps, model_type = model_type)
                eoptimal_fit_dict[subj_name + '_' + model_type + '_first']  = fit_static_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], train_recursive_p, action_eps = action_eps, model_type = model_type)
                eoptimal_fit_dict[subj_name + '_' + model_type + '_second']  = fit_static_model(train_ts_dis, test_dfa.iloc[df_midpoint:], train_recursive_p, action_eps = action_eps, model_type = model_type)
            if subj_name + '_' + model_type + '_first' not in ignore_fit_dict.keys():                
                ignore_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_static_model(train_ts_dis, test_dfa, .5, action_eps = action_eps, model_type = model_type)                
                ignore_fit_dict[subj_name + '_' + model_type + '_first']  = fit_static_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], .5, action_eps = action_eps, model_type = model_type)
                ignore_fit_dict[subj_name + '_' + model_type + '_second']  = fit_static_model(train_ts_dis, test_dfa.iloc[df_midpoint:], .5, action_eps = action_eps, model_type = model_type)
            if subj_name + '_first' not in midline_fit_dict.keys():               
                midline_fit_dict[subj_name + '_fullRun'] = fit_midline_model(test_dfa)                
                midline_fit_dict[subj_name + '_first'] = fit_midline_model(test_dfa.iloc[0:df_midpoint])
                midline_fit_dict[subj_name + '_second'] = fit_midline_model(test_dfa.iloc[df_midpoint:])
            if subj_name + '_first' not in switch_fit_dict.keys():    
                switch_fit_dict[subj_name + '_fullRun'] = fit_switch_model(test_dfa)                
                switch_fit_dict[subj_name + '_first'] = fit_switch_model(test_dfa.iloc[0:df_midpoint])
                switch_fit_dict[subj_name + '_second'] = fit_switch_model(test_dfa.iloc[df_midpoint:])
        for model_type in ['TS']:
            print(model_type)
            if subj_name + '_' + model_type + '_first' not in memory_fit_dict.keys(): 
                memory_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_memory_model(train_ts_dis, test_dfa, perseverance = 0)                
                memory_fit_dict[subj_name + '_' + model_type + '_first'] = fit_memory_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], perseverance = 0)
                memory_fit_dict[subj_name + '_' + model_type+ '_second'] = fit_memory_model(train_ts_dis, test_dfa.iloc[df_midpoint:], perseverance = 0)
            if subj_name + '_' + model_type + '_first' not in perseverance_fit_dict.keys(): 
                perseverance_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_memory_model(train_ts_dis, test_dfa, k = 0)    
                perseverance_fit_dict[subj_name + '_' + model_type + '_first'] = fit_memory_model(train_ts_dis, test_dfa.iloc[0:df_midpoint], k = 0)
                perseverance_fit_dict[subj_name + '_' + model_type + '_second'] = fit_memory_model(train_ts_dis, test_dfa.iloc[df_midpoint:], k = 0)
            if subj_name + '_' + model_type + '_first' not in permem_fit_dict.keys(): 
                permem_fit_dict[subj_name + '_' + model_type + '_fullRun'] = fit_memory_model(train_ts_dis, test_dfa)    
                permem_fit_dict[subj_name + '_' + model_type + '_first'] = fit_memory_model(train_ts_dis, test_dfa.iloc[0:df_midpoint])
                permem_fit_dict[subj_name + '_' + model_type + '_second'] = fit_memory_model(train_ts_dis, test_dfa.iloc[df_midpoint:])
            
        
        # *********************************************
        # Set up observers
        # *********************************************
        
        # **************TRAIN*********************
        # This observer know the exact statistics of the task, always chooses correctly
        # given that it chooses the correct task-set, and perfectly learns from feedback.
        # This means that it sets the prior probability for each ts to the transition probabilities
        # of the correct task-set on each trial (which a subject 'could' do due to the
        # deterministic feedback). Basically, after receiving FB, the ideal observer
        # knows exactly what task it is in and should act accordingly.
        observer_prior = [.5,.5]
        posteriors = []
        for i,trial in train_dfa.iterrows():
            c = trial.context
            posteriors.append(calc_posterior(c,observer_prior,ts_dis))
            ts = trial.ts
            observer_prior = np.round([.9*(1-ts)+.1*ts,.9*ts+.1*(1-ts)],2)
        train_dfa.loc[:,'optimal_posterior'] = posteriors

        
        # Optimal observer for train, without feedback     
        no_fb = BiasPredModel(ts_dis, [.5,.5], r1 = .9, r2 = .9, TS_eps = 0, action_eps = 0)
        observer_choices = []
        posteriors = []
        for i,trial in train_dfa.iterrows():
            c = trial.context
            posteriors.append(no_fb.calc_posterior(c)[1])
        posteriors = np.array(posteriors)
        train_dfa['no_fb_posterior'] = posteriors
        
        
        # **************TEST*********************
        model_type = 'TS'
        
        for p in ['_first','_second','_fullRun']:
            # Bias2 observer for test    
            params = bias2_fit_dict[subj_name + '_' + model_type + p]
            bias2 = BiasPredModel(train_ts_dis, [.5,.5],**params)
            params = bias1_fit_dict[subj_name + '_' + model_type + p]
            bias1 = BiasPredModel(train_ts_dis, [.5,.5], **params)
            params = eoptimal_fit_dict[subj_name + '_' + model_type + p]
            eoptimal = BiasPredModel(train_ts_dis, [.5,.5], **params)
            params = ignore_fit_dict[subj_name + '_' + model_type + p]
            ignore = BiasPredModel(train_ts_dis, [.5,.5], **params)
            params = memory_fit_dict[subj_name + '_' + model_type + p]
            memory = MemoryModel(train_ts_dis, **params)
            params = perseverance_fit_dict[subj_name + '_' + model_type + p]
            perseverance = MemoryModel(train_ts_dis,  **params)
            params = permem_fit_dict[subj_name + '_' + model_type + p]
            permem = MemoryModel(train_ts_dis,  **params)
            if p != '_fullRun':
                postfix = p
            else:
                postfix = ''
                
            # Fit observer for test        
            gen_bias_TS_posteriors([bias2, bias1, eoptimal, ignore], test_dfa, ['bias2', 'bias1', 'eoptimal', 'ignore'], postfix = postfix)
            gen_memory_TS_posteriors([memory, perseverance, permem], test_dfa, ['memory', 'perseverance', 'permem'], postfix = postfix)
        
        for model in ['bias2', 'bias1', 'eoptimal', 'ignore', 'memory', 'perseverance','permem']:
            cross_posteriors = pd.concat([test_dfa[:df_midpoint][model + '_posterior_second'],test_dfa[df_midpoint:][model + '_posterior_first']])
            test_dfa[model + '_posterior_cross'] = cross_posteriors
            test_dfa.drop([model + '_posterior_first', model + '_posterior_second'], inplace = True, axis = 1)
        
        
        # midline observer for test  
        eps = midline_fit_dict[subj_name + '_fullRun']['eps']
        eps_first = midline_fit_dict[subj_name + '_first']['eps']
        eps_second = midline_fit_dict[subj_name + '_second']['eps']
        posteriors = []
        posteriors_cross = []
        for i,trial in test_dfa.iterrows():
            c = max(0,np.sign(trial.context))
            posteriors.append(abs(c - eps))
            if i<df_midpoint:
                posteriors_cross.append(abs(c-eps_second))
            else:
                posteriors_cross.append(abs(c-eps_first))
        posteriors = np.array(posteriors)
        posteriors_cross = np.array(posteriors_cross)        
        test_dfa['midline_posterior'] = posteriors
        test_dfa['midline_posterior_cross'] = posteriors_cross
    
        # Switch observer for test  
        params = switch_fit_dict[subj_name + '_fullRun']      
        switch = SwitchModel(**params)
        params = switch_fit_dict[subj_name + '_first']      
        switch_first = SwitchModel(**params)
        params = switch_fit_dict[subj_name + '_second']      
        switch_second = SwitchModel(**params)
        posteriors = []
        posteriors_cross = []
        for i,trial in test_dfa.iterrows():
            if i == 0:
                 last_choice = -1 
            else:
                last_choice = test_dfa.subj_ts[i-1]
            trial_choice = trial.subj_ts
            conf = switch.calc_TS_prob(last_choice)
            if i<df_midpoint:
                conf_cross = switch_second.calc_TS_prob(last_choice)
            else:
                conf_cross = switch_first.calc_TS_prob(last_choice)
            posteriors.append(conf[trial_choice])  
            posteriors_cross.append(conf_cross[trial_choice])
        posteriors = np.array(posteriors)
        posteriors_cross = np.array(posteriors_cross) 
        test_dfa['switch_posterior'] = posteriors
        test_dfa['switch_posterior_cross'] = posteriors_cross
        
        #test_dfa['bias2_choices'] = (posteriors>.5).astype(int)
        #test_dfa['bias2_switch'] = (test_dfa.bias2_posterior>.5).diff()
    
        train_dfa['id'] = subj_name
        test_dfa['id'] = subj_name
        gtrain_df = pd.concat([gtrain_df,train_dfa])
        gtest_df = pd.concat([gtest_df,test_dfa])   
        gtaskinfo.append(taskinfo)
    
     
    gtaskinfo = pd.DataFrame(gtaskinfo)


            
# *********************************************
# Exclusion criterion
# ********************************************* 
    
    #arbitrary behavioral exclusion
    # Exclude subjects where stim_confom is below some threshold 
    select_ids = gtest_df.groupby('id').mean().stim_conform>.75
    select_ids = select_ids[select_ids]
    select_rows = [i in select_ids for i in gtrain_df.id]
    gtrain_conform_df = gtrain_df[select_rows]
    select_rows = [i in select_ids for i in gtest_df.id]
    gtest_conform_df = gtest_df[select_rows]
    ids = select_ids.index
    
    # separate learner group
    select_ids = gtest_conform_df.groupby('id').mean().correct > .5
    select_ids = select_ids[select_ids]
    select_rows = [i in select_ids for i in gtrain_conform_df.id]
    gtrain_learn_df = gtrain_conform_df[select_rows]
    select_rows = [i in select_ids for i in gtest_conform_df.id]
    gtest_learn_df = gtest_conform_df[select_rows]
    learn_ids = select_ids.index

# *********************************************
# Save
# ********************************************* 

    pickle.dump(bias2_fit_dict,open('Analysis_Output/bias2_parameter_fits.p','wb'), protocol=2)
    pickle.dump(bias1_fit_dict,open('Analysis_Output/bias1_parameter_fits.p','wb'), protocol=2)
    pickle.dump(eoptimal_fit_dict,open('Analysis_Output/eoptimal_parameter_fits.p','wb'), protocol=2)
    pickle.dump(ignore_fit_dict,open('Analysis_Output/ignore_parameter_fits.p','wb'), protocol=2)
    pickle.dump(midline_fit_dict,open('Analysis_Output/midline_parameter_fits.p','wb'), protocol=2)
    pickle.dump(switch_fit_dict,open('Analysis_Output/switch_parameter_fits.p','wb'), protocol=2)
    pickle.dump(memory_fit_dict,open('Analysis_Output/memory_parameter_fits.p','wb'), protocol=2)
    pickle.dump(perseverance_fit_dict,open('Analysis_Output/perseverance_parameter_fits.p','wb'), protocol=2)
    pickle.dump(permem_fit_dict,open('Analysis_Output/permem_parameter_fits.p','wb'), protocol=2)
    gtest_learn_df.to_pickle('Analysis_Output/gtest_learn_df.pkl')
    gtest_conform_df.to_pickle('Analysis_Output/gtest_conform_df.pkl')
    gtest_df.to_pickle('Analysis_Output/gtest_df.pkl')
    gtrain_learn_df.to_pickle('Analysis_Output/gtrain_learn_df.pkl')
    gtrain_conform_df.to_pickle('Analysis_Output/gtrain_conform_df.pkl')
    gtrain_df.to_pickle('Analysis_Output/gtrain_df.pkl')
    gtaskinfo.to_pickle = ('Analysis_Output_gtaskinfo.pkl')




