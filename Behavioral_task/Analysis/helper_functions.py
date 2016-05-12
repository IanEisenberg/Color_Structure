# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:22:54 2015

@author: Ian
"""

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab, lmfit
import random as r
from helper_classes import BiasPredModel, SwitchModel, MemoryModel

def track_runs(iterable):
    """
    Return the item with the most consecutive repetitions in `iterable`.
    If there are multiple such items, return the first one.
    If `iterable` is empty, return `None`.
    """
    track_repeats=[]
    current_element = None
    current_repeats = 0
    element_i = 0
    for element in iterable:
        if current_element == element:
            current_repeats += 1
        else:
            track_repeats.append((current_repeats,current_element, element_i-current_repeats))
            current_element = element
            current_repeats = 1
        element_i += 1
    track_repeats = track_repeats[1:]
    return track_repeats

def bar(x, y, title):
    plot = plt.bar(x,y,width = .5)
    plt.title(str(title))
    return plot

def softmax(probs, inv_temp):
    return np.exp(probs*inv_temp)/sum(np.exp(probs*inv_temp))

def calc_posterior(data,prior,likelihood_dist):
    n = len(prior)
    likelihood = [dis.pdf(data) for dis in likelihood_dist]
    numer = np.array([likelihood[i] * prior[i] for i in range(n)])
    try:
        dinom = [np.sum(list(zip(*numer))[i]) for i in range(len(numer[0]))]
    except TypeError:
        dinom = np.sum(numer)
    posterior = numer/dinom
    return posterior

def genSeq(l,p):
    seq = [1]
    while abs(np.mean(seq)-.5) > .1:
        curr_state = round(r.random())
        state_reps = 0
        seq = []
        for _ in range(l-1):
            seq.append(curr_state)
            if r.random() > p or state_reps > 25:
                curr_state = 1-curr_state
                state_reps = 0
            else:
                state_reps += 1
    return seq
                
def seqStats(l,p,reps):
    seqs=[]
    for _ in range(reps):
        tmp = genSeq(l,p)
        for i in track_runs(tmp):
            seqs.append(i[0])
    return (np.mean(seqs), np.std(seqs))

def genExperimentSeq(l, p, ts_dis):
    """Generates an experiment seq
    """
    bin_boundaries = np.linspace(-1,1,11)
    seq = genSeq(l,p)
    context = []
    for s in seq:
        binned = -1.1 + np.digitize([ts_dis[s].rvs()],bin_boundaries)*.2
        truncated_context = round(max(-1, min(1, binned[0])),2)
        context.append(truncated_context)
    df = pd.DataFrame({'context': pd.Series(context), 'ts': pd.Series(seq)})
    return df

def simulateModel(model, ts_dis, model_name = 'model', l = 800, p = .9, mode = 'e-greedy'):
    seq = genExperimentSeq(l,p,ts_dis)
    posteriors = []
    choices = []
    for c in seq['context']:
        posteriors.append(model.calc_posterior(c))
        choices.append(model.choose(mode = mode))
    seq['subj_ts'] = choices
    seq['posteriors'] = posteriors
    seq['model'] = model_name
    return seq
          
    
def preproc_data(traindata, testdata, taskinfo):
            """ Sets TS2 to always be associated with the 'top' of the screen (positive context values),
            creates a log_rt column and outputs task statistics during training
            :return: train_ts_dis, train_recursive_p, action_eps
            """
            #flip contexts if necessary
            states = taskinfo['states']
            tasksets = {val['ts']: {'c_mean': val['c_mean'], 'c_sd': val['c_sd']} for val in states.values()}
            ts2_side = np.sign(tasksets[1]['c_mean'])
            traindata['true_context'] = traindata['context']
            testdata['true_context'] = testdata['context']            
            traindata['context']*=ts2_side
            testdata['context']*=ts2_side
            #add log rt columns
            traindata['log_rt'] = np.log(traindata.rt)
            testdata['log_rt'] = np.log(testdata.rt)
            # What was the mean contextual value for each taskset during this train run?
            train_ts_means = list(traindata.groupby('ts').agg(np.mean).context)
            # Same for standard deviation
            train_ts_std = list(traindata.groupby('ts').agg(np.std).context)
            train_ts_dis = [norm(m, s) for m, s in zip(train_ts_means, train_ts_std)]
            train_recursive_p = 1 - traindata.switch.mean()
            action_eps = 1-np.mean([testdata['response'][i] in testdata['stim'][i] for i in testdata.index])
            return train_ts_dis, train_recursive_p, action_eps

#*********************************************
# Model fitting functions
#*********************************************

def fit_bias2_model(train_ts_dis, data, init_prior = [.5,.5], action_eps = 0, model_type = 'action', print_out = True, return_out = False):
    """
    Function to fit parameters to the bias2 model (fit r1, r2 and epsilon).
    Model can either fit to TS choices or actions
    """
    def errfunc(params,df):
        r1 = params['r1']
        r2 = params['r2']
        eps = params['TS_eps']
    
        init_prior = [.5,.5]
        model = BiasPredModel(train_ts_dis, init_prior, r1=r1, r2=r2, TS_eps=eps, action_eps = action_eps)
        model_likelihoods = []
        for i in df.index:
            c = df.context[i]
            TS_choice = df.subj_ts[i]
            if model_type == 'TS':
                TS_probs = model.calc_posterior(c)
                model_likelihoods.append(TS_probs[TS_choice])
            elif model_type == 'action':
                s = df.stim[i]
                response_choice = df.response[i]
                action_probs = model.calc_action_posterior(s,c)
                model_likelihoods.append(action_probs[response_choice])
        # minimize
        return abs(np.sum(np.log(np.array(model_likelihoods)))) # single value
    
    # Fit bias model
    fit_params = lmfit.Parameters()
    fit_params.add('r1', value=.5, min=0, max=1)
    fit_params.add('r2', value=.5, min=0, max=1)
    fit_params.add('TS_eps', value=.1, min=0, max=1)
    out = lmfit.minimize(errfunc, fit_params, method = 'lbfgsb', kws={'df': data})
    if print_out:
        lmfit.report_fit(out)
    if return_out:
        return out
    else:
        return out.params.valuesdict()
    
def fit_bias1_model(train_ts_dis, data, init_prior = [.5,.5], action_eps = 0, model_type = 'action', print_out = True, return_out = False):
    """
    Function to fit parameters to the bias2 model (fit r and epsilon)
    """
    def errfunc(params,df):
        r1 = params['rp']
        r2 = params['rp']
        eps = params['TS_eps']

        init_prior = [.5,.5]
        model = BiasPredModel(train_ts_dis, init_prior, r1=r1, r2=r2, TS_eps=eps, action_eps = action_eps)
        model_likelihoods = []
        for i in df.index:
            c = df.context[i]
            TS_choice = df.subj_ts[i]
            if model_type == 'TS':
                TS_probs = model.calc_posterior(c)
                model_likelihoods.append(TS_probs[TS_choice])
            elif model_type == 'action':
                s = df.stim[i]
                response_choice = df.response[i]
                action_probs = model.calc_action_posterior(s,c)
                model_likelihoods.append(action_probs[response_choice])
        # minimize
        return abs(np.sum(np.log(np.array(model_likelihoods)))) # single value
    
    # Fit bias model
    fit_params = lmfit.Parameters()
    fit_params.add('rp', value=.5, min=0, max=1)
    fit_params.add('TS_eps', value = .1, min=0, max=1)
    out = lmfit.minimize(errfunc, fit_params, method = 'lbfgsb', kws={'df': data})
    if print_out:
        lmfit.report_fit(out)
    if return_out:
        return out
    else:
        return out.params.valuesdict()
    
def fit_static_model(train_ts_dis, data, r_value, init_prior = [.5,.5], action_eps = 0, model_type = 'action', print_out = True, return_out = False):
    """
    Function to fit any model where recursive probabilities are fixed, like an
    optimal model (r1=r2=.9) or a base-rate neglect model (r1=r2=.5)
    """
    def errfunc(params,df):
        eps = params['TS_eps']

        init_prior = [.5,.5]
        model = BiasPredModel(train_ts_dis, init_prior, rp = r_value, TS_eps=eps, action_eps = action_eps)
        model_likelihoods = []
        for i in df.index:
            c = df.context[i]
            TS_choice = df.subj_ts[i]
            if model_type == 'TS':
                TS_probs = model.calc_posterior(c)
                model_likelihoods.append(TS_probs[TS_choice])
            elif model_type == 'action':
                s = df.stim[i]
                response_choice = df.response[i]
                action_probs = model.calc_action_posterior(s,c)
                model_likelihoods.append(action_probs[response_choice])
        # minimize
        return abs(np.sum(np.log(np.array(model_likelihoods)))) # single value
    
    # Fit bias model
    fit_params = lmfit.Parameters()
    fit_params.add('TS_eps', value = .1, min=0, max=1)
    out = lmfit.minimize(errfunc, fit_params, method = 'lbfgsb', kws={'df': data})
    if print_out:
        lmfit.report_fit(out)
    fit_params = out.params.valuesdict()
    fit_params['rp'] = r_value
    if return_out:
        return out
    else:
        return fit_params

def fit_midline_model(data, print_out = True, return_out = False):
    def midline_errfunc(params,df):
        eps = params['eps'].value
        context_sgn = np.array([max(i,0) for i in df.context_sign])
        choice = df.subj_ts
        #minimize
        return -np.sum(np.log(abs(abs(choice - (1-context_sgn))-eps)))

    #Fit bias model
    #attempt to simplify:
    fit_params = lmfit.Parameters()
    fit_params.add('eps', value = .1, min = 0, max = 1)
    out = lmfit.minimize(midline_errfunc,fit_params, method = 'lbfgsb', kws= {'df': data})
    if print_out:
        lmfit.report_fit(out)
    if return_out:
        return out
    else:
        return out.params.valuesdict()
    
def fit_switch_model(data, print_out = True, return_out = False):
    def switch_errfunc(params,df):
        params = params.valuesdict()
        r1 = params['r1']
        r2 = params['r2']   
        eps = params['eps']
        model = SwitchModel(r1 = r1, r2 = r2, eps = eps)
        model_likelihoods = []
        model_likelihoods.append(.5)
        for i in df.index[1:]:
            last_choice = df.subj_ts[i-1]
            trial_choice = df.subj_ts[i]
            conf = model.calc_TS_prob(last_choice)
            model_likelihoods.append(conf[trial_choice])
            
        # minimize
        return abs(np.sum(np.log(model_likelihoods))) # single value
    # Fit switch model
    fit_params = lmfit.Parameters()
    fit_params.add('r1', value=.5, min=0, max=1)
    fit_params.add('r2', value=.5, min=0, max=1)
    fit_params.add('eps', value = .1, min = 0, max = 1)
    out = lmfit.minimize(switch_errfunc,fit_params, method = 'lbfgsb', kws= {'df': data})
    if print_out:
        lmfit.report_fit(out)
    if return_out:
        return out
    else:
        return out.params.valuesdict()

def fit_memory_model(train_ts_dis, data, k = None, perseverance = None, print_out = True, return_out = False):
    def errfunc(params,df):
        params = params.valuesdict()
        k = params['k']
        perseverance = params['perseverance']   
        bias = params['bias']
        TS_eps = params['TS_eps']
        model = MemoryModel(train_ts_dis, k = k, perseverance = perseverance, bias = bias, TS_eps=TS_eps)
        model_likelihoods = []
        model_likelihoods.append(.5)
        for i in df.index[1:]:
            last_choice = df.subj_ts[i-1]
            trial_choice = df.subj_ts[i]
            c = df.context[i]
            conf = model.calc_posterior(c, last_choice)
            model_likelihoods.append(conf[trial_choice])
        # minimize
        return abs(np.sum(np.log(np.array(model_likelihoods)))) # single value
    # Fit memory model
    fit_params = lmfit.Parameters()
    if k == None:
        fit_params.add('k', value=1)
    else:
        fit_params.add('k', value = k, vary = False)
    if perseverance == None:
        fit_params.add('perseverance', value=.5, min=0, max=1)
    else:
        fit_params.add('perseverance', value = perseverance, vary = False)
    fit_params.add('bias', value = .5, min = 0, max = 1)
    fit_params.add('TS_eps', value = .1, min = 0, max = 1)
    out = lmfit.minimize(errfunc,fit_params, method = 'lbfgsb', kws= {'df': data})
    fit_params = out.params.valuesdict()
    if print_out:
        lmfit.report_fit(out)
    if return_out:
        return out
    else:
        return fit_params
    
    
#*********************************************
# Generate Model Predtions
#*********************************************

def gen_bias_TS_posteriors(models, data, model_names = None, model_type = 'TS', reduce = True, get_choice = False, postfix = ''):
    """ Generates an array of TS or model(s)
    :model: model or array of models that has a calc_posterior method
    :data: dataframe with a context
    :reduce: bool, if True only show the posterior for task-set 2
    """
    assert len(model_names)
    if not isinstance(models,list):
        models = [models]
    if model_names:
        if not isinstance(model_names,list):
            model_names = [model_names]
        assert len(model_names) == len(models), \
            'Model_names must be the same length as models'
    model_posteriors = [[] for _ in  range(len(models))]
    model_choices = [[] for _ in  range(len(models))]
    for i,trial in data.iterrows():
        c = trial.context
        s = trial.stim
        for j,model in enumerate(models):
            if model_type == 'TS':
                posterior = model.calc_posterior(c)
            elif model_type == 'action':
                posterior = model.calc_action_posterior(s,c)
            if reduce:
                model_posteriors[j].append(posterior[1])
            else:
                model_posteriors[j].append(posterior)
            if get_choice:
                model_choices[j].append([model.choose() for _ in range(10)])
    for j,posteriors in enumerate(model_posteriors):
        if model_names:
            model_name = model_names[j]
        else:
            model_name = 'model_%s' % j
        data[model_name + '_posterior' + postfix] = model_posteriors[j]
        if get_choice:
            data[model_name + '_choices' + postfix] = model_choices[j]
        

def gen_memory_TS_posteriors(models, data, model_names = None, model_type = 'TS', reduce = True, get_choice = False, postfix = ''):
    """ Generates an array of TS or model(s)
    :model: model or array of models that has a calc_posterior method
    :data: dataframe with a context
    :reduce: bool, if True only show the posterior for task-set 2
    """
    assert len(model_names)
    if not isinstance(models,list):
        models = [models]
    if model_names:
        if not isinstance(model_names,list):
            model_names = [model_names]
        assert len(model_names) == len(models), \
            'Model_names must be the same length as models'
    model_posteriors = [[] for _ in  range(len(models))]
    model_choices = [[] for _ in  range(len(models))]
    last_choice = None
    for i,trial in data.iterrows():
        c = trial.context
        s = trial.stim
        for j,model in enumerate(models):
            if model_type == 'TS':
                posterior = model.calc_posterior(c, last_choice)
            elif model_type == 'action':
                posterior = model.calc_action_posterior(s,c)
            if reduce:
                model_posteriors[j].append(posterior[1])
            else:
                model_posteriors[j].append(posterior)
            if get_choice:
                model_choices[j].append([model.choose() for _ in range(10)])
        last_choice = trial.subj_ts
            
    for j,posteriors in enumerate(model_posteriors):
        if model_names:
            model_name = model_names[j]
        else:
            model_name = 'model_%s' % j
        data[model_name + '_posterior' + postfix] = model_posteriors[j]
        if get_choice:
            data[model_name + '_choices' + postfix] = model_choices[j]
        
#*********************************************
# Plotting
#*********************************************

def plot_run(sub,plotting_dict, exclude = [], fontsize = 16):
    #plot the posterior estimates for different models, the TS they currently select
    #and the vertical position of the stimulus
    sns.set_style("white")
    plt.hold(True)
    models = []
    displacement = 0
    #plot model certainty and task-set choices
    for arg in plotting_dict.values():
        if arg[2] not in exclude:
            plt.plot(sub.trial_count,sub[arg[0]]*2,arg[1], label = arg[2], lw = 2)
            plt.plot(sub.trial_count, [int(val>.5)+3+displacement for val in sub[arg[0]]],arg[1]+'o')
            displacement+=.15
            models.append(arg[0])
    plt.axhline(1, color = 'y', ls = 'dashed', lw = 2)
    plt.axhline(2.5, color = 'k', ls = 'dashed', lw = 3)
    #plot subject choices (con_shape = conforming to TS1)
    #plot current TS, flipping bit to plot correctly
    plt.plot(sub.trial_count,(sub.ts)-2, 'go', label = 'operating TS')
    plt.plot(sub.trial_count, sub.context/2-1.5,'k', lw = 2, label = 'stimulus height')
    plt.plot(sub.trial_count, sub.con_2dim+2.85, 'yo', label = 'subject choice')
    plt.yticks([-2, -1.5, -1, 0, 1, 2, 3.1, 4.1], [ -1, 0 , 1,'0%', '50%',  '100%', 'TS2 Choice', 'TS1 Choice'], size = fontsize-4 )
    plt.xticks(size = fontsize - 4)
    plt.xlim([min(sub.index)-.5,max(sub.index)])
    plt.ylim(-2.5,5)
    #subdivide graph
    plt.axhline(-.5, color = 'k', ls = 'dashed', lw = 3)
    plt.axhline(-1.5, color = 'y', ls = 'dashed', lw = 2)
    #axes labels
    plt.xlabel('Trial Number', size = fontsize, fontweight = 'bold')
    plt.ylabel('Predicted P(TS1)', size = fontsize, fontweight = 'bold')
    ax = plt.gca()
    ax.yaxis.set_label_coords(-.1, .45)
    pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08),
              ncol=3, fancybox=True, shadow=True, prop={'size':fontsize}, frameon = True)
