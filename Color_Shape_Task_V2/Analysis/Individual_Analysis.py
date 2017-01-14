"""
Created on Mon Apr 27 11:16:08 2015

@author: Ian
"""
import numpy as np
from Load_Data import load_data, preproc_data
from helper_classes import BiasPredModel, SwitchModel, MemoryModel
from helper_functions import fit_bias2_model, fit_bias1_model, fit_static_model, \
    fit_switch_model, fit_midline_model, fit_memory_model, calc_posterior, gen_bias_TS_posteriors, \
    gen_memory_TS_posteriors
import pickle, glob, re
import pandas as pd
from scipy.stats import norm
import warnings

# Suppress runtimewarning due to pandas bug
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# *********************************************
# Helper functions
# ********************************************
def get_data(filey):
    file_name = re.match(r'.*/RawData.(.*).yaml', filey).group(1)
    try:
        info = pickle.load(open('../Data/' + file_name + '.pkl', 'rb'))
        taskinfo, dfa = [info.get(k) for k in ['taskinfo', 'dfa']]
    except IOError:
        taskinfo, df, dfa = load_data(filey, file_name, mode='train')
        info = {'taskinfo': taskinfo, 'dfa': dfa}
        pickle.dump(info, open('../Data/' + file_name + '.pkl','wb'), protocol = 2)
    return taskinfo,  dfa

def get_model_dicts(filey):
    try:
        model_dicts = pickle.load(open(filey,'r'))
    except IOError:
        model_dicts = {}
        model_names = ['bias2','bias1','eoptimal','ignore','midline','switch','memory','perseverance','permem']
        # set up dictionaries to hold fitted parameters for each subject
        fitting_functions = [fit_bias2_model, fit_bias1_model, fit_static_model, fit_static_model, fit_midline_model, fit_switch_model, fit_memory_model, fit_memory_model, fit_memory_model]
        for name, fun in zip(model_names,fitting_functions):
            model = {'fitting_fun': fun}
            try:
                model['fit_dict'] = pickle.load(open('Analysis_Output/' + name + '_parameter_fits.pkl', 'rb'))
            except:
                model['fit_dict'] = {}
            model_dicts[name] = model    
    return model_dicts
    
def fit_test_models(ts_distributions, rp, action_eps, test, model_dicts, verbose = True):
    df_midpoint = int(len(test)/2)
    for model_type in ['TS']:
        for name, model in model_dicts.items():
            if verbose:
                print('\nFitting %s Model\n' % name)
            fit_dict = model['fit_dict']
            fun = model['fitting_fun']
            args = {}
            if name in ['bias2', 'bias1']:
                args = {'action_eps': action_eps, 'model_type': model_type}
            elif name == 'eoptimal':
                args = {'rp': rp, 'action_eps': action_eps, 'model_type': model_type}
            elif name == 'ignore':
                args = {'rp': .5, 'action_eps': action_eps, 'model_type': model_type}
            elif name == 'memory':
                args = {'perseverance': 0}
            elif name == 'perseverance':
                args = {'k': 0}
            if subj_name + '_' + model_type + '_first' not in fit_dict.keys():
                if name in ['midline', 'switch']:
                    fit_dict[subj_name + '_' + model_type + '_fullRun'] = fun(test, verbose = verbose)
                    fit_dict[subj_name + '_' + model_type + '_first'] = fun(test.iloc[0:df_midpoint], verbose = verbose)
                    fit_dict[subj_name + '_' + model_type + '_second'] = fun(test.iloc[0:df_midpoint], verbose = verbose)
                else:
                    fit_dict[subj_name + '_' + model_type + '_fullRun'] = fun(ts_distributions, test, verbose = verbose, **args)
                    fit_dict[subj_name + '_' + model_type + '_first'] = fun(ts_distributions, test.iloc[0:df_midpoint], verbose = verbose, **args)
                    fit_dict[subj_name + '_' + model_type + '_second'] = fun(ts_distributions, test.iloc[0:df_midpoint], verbose = verbose, **args)



# *********************************************
# Set up defaults
# *********************************************
plot = False
save = True
model_dicts = get_model_dicts('Analysis_Output/model_fits.pkl')

# *********************************************
# Load Data
# ********************************************
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
    data_files = sorted(glob.glob('../Data/RawData/*yaml'))
    train_files = [f for f in data_files if 'test' not in f]
    test_files = [f for f in data_files if 'test' in f]

    for train_file, test_file in zip(train_files, test_files):
        subj_name = re.match(r'.*/RawData.(\w*)_Prob*', test_file).group(1)
        print(subj_name)
    
        # load pickled data. If pickled data doesn't exist, pickle the data
        taskinfo, train_dfa = get_data(train_file)
        taskinfo, test_dfa = get_data(test_file)

        # *********************************************
        # Preliminary Setup
        # *********************************************
        ts_dis = [norm(**taskinfo['states'][s]['dist_args']) for s in [0,1]]
        train_ts_dis,train_recursive_p,action_eps = preproc_data(train_dfa,test_dfa,taskinfo)        
        
        # *********************************************
        # Model fitting
        # *********************************************
        fit_test_models(train_ts_dis, train_recursive_p, action_eps, test_dfa, model_dicts)
        
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

        
        # **************TEST*********************
        model_type = '_TS'
        for p in ['_first','_second','_fullRun']:
            bias_models = []
            memory_models = []
            # Bias2 observer for test    
            for model in ['bias2', 'bias1', 'eoptimal', 'ignore']:
                params = model_dicts[model]['fit_dict'][subj_name + model_type + p]
                bias_models.append(BiasPredModel(train_ts_dis, [.5,.5], **params)) 
                
            for model in ['memory','perseverance','permem']:
                params = model_dicts[model]['fit_dict'][subj_name + model_type + p]
                memory_models.append(MemoryModel(train_ts_dis, **params)) 
                
            if p != '_fullRun':
                postfix = p
            else:
                postfix = ''
                
            # Fit observer for test        
            gen_bias_TS_posteriors(bias_models, test_dfa, ['bias2', 'bias1', 'eoptimal', 'ignore'], postfix = postfix)
            gen_memory_TS_posteriors(memory_models, test_dfa, ['memory', 'perseverance', 'permem'], postfix = postfix)
        
        for model in ['bias2', 'bias1', 'eoptimal', 'ignore', 'memory', 'perseverance','permem']:
            df_midpoint = int(len(test_dfa)/2)
            cross_posteriors = pd.concat([test_dfa[:df_midpoint][model + '_posterior_second'],test_dfa[df_midpoint:][model + '_posterior_first']])
            test_dfa[model + '_posterior_cross'] = cross_posteriors
            test_dfa.drop([model + '_posterior_first', model + '_posterior_second'], inplace = True, axis = 1)
        
        midline_eps = []
        switch_models = []
        for p in ['_first','_second','_fullRun']:
            midline_eps.append(model_dicts['midline']['fit_dict'][subj_name + model_type + p]['eps'])
            switch_params = model_dicts['switch']['fit_dict'][subj_name + model_type + p]          
            switch_models.append(SwitchModel(**switch_params))
                                  
        # midline observer for test  
        posteriors = []
        posteriors_cross = []
        for i,trial in test_dfa.iterrows():
            c = max(0,np.sign(trial.context))
            posteriors.append(abs(c - midline_eps[2]))
            if i<df_midpoint:
                posteriors_cross.append(abs(c-midline_eps[1]))
            else:
                posteriors_cross.append(abs(c-midline_eps[0]))
        posteriors = np.array(posteriors)
        posteriors_cross = np.array(posteriors_cross)        
        test_dfa['midline_posterior'] = posteriors
        test_dfa['midline_posterior_cross'] = posteriors_cross
    
        # Switch observer for test  
        posteriors = []
        posteriors_cross = []
        for i,trial in test_dfa.iterrows():
            if i == 0:
                 last_choice = -1 
            else:
                last_choice = test_dfa.subj_ts[i-1]
            trial_choice = trial.subj_ts
            conf = switch_models[2].calc_TS_prob(last_choice)
            if i<df_midpoint:
                conf_cross = switch_models[1].calc_TS_prob(last_choice)
            else:
                conf_cross = switch_models[0].calc_TS_prob(last_choice)
            posteriors.append(conf[trial_choice])  
            posteriors_cross.append(conf_cross[trial_choice])
        posteriors = np.array(posteriors)
        posteriors_cross = np.array(posteriors_cross) 
        test_dfa['switch_posterior'] = posteriors
        test_dfa['switch_posterior_cross'] = posteriors_cross
        
        # ********************************************************************
        # Clean up and add df to group df
        # ********************************************************************
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

    pickle.dump(model_dicts,open('Analysis_Output/model_fits.pkl','wb'), protocol=2)
    pickle.dump(gtest_learn_df, open('Analysis_Output/gtest_learn_df.pkl','wb'), protocol=2)
    pickle.dump(gtest_conform_df, open('Analysis_Output/gtest_conform_df.pkl','wb'), protocol=2)
    pickle.dump(gtest_df, open('Analysis_Output/gtest_df.pkl','wb'), protocol=2)
    pickle.dump(gtrain_learn_df, open('Analysis_Output/gtrain_learn_df.pkl','wb'), protocol=2)
    pickle.dump(gtrain_conform_df, open('Analysis_Output/gtrain_conform_df.pkl','wb'), protocol=2)
    pickle.dump(gtrain_df, open('Analysis_Output/gtrain_df.pkl','wb'), protocol=2)
    pickle.dump(gtaskinfo, open('Analysis_Output_gtaskinfo.pkl','wb'), protocol=2)




