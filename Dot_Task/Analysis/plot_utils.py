import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psychopy.data import FitCumNormal, FitWeibull
import seaborn as sns
from Dot_Task.Analysis.load_data import load_threshold_data
from Dot_Task.Analysis.utils import fit_choice_fun, fit_response_fun

def beautify_legend(legend, colors, fontsize=None):
    for i, text in enumerate(legend.get_texts()):
        text.set_color(colors[i])
    for item in legend.legendHandles:
        item.set_visible(False)
    legend.get_frame().set_linewidth(0.0)
    if fontsize:
        plt.setp(legend.get_texts(), fontsize=fontsize)

def plot_choice_fun(clf, minval, maxval, ax=None, plot_kws={}):
    if ax is None:
        f, ax = plt.subplots()
    X = np.linspace(minval, maxval, 100)
    y = [i[1] for i in clf.predict_proba(X.reshape(-1,1))]
    ax.plot(X, y, **plot_kws)
    
def plot_response_fun(responseFun, ax=None, plot_kws={}):
    if ax is None:
        f, ax = plt.subplots()
    X = np.linspace(0,responseFun.inverse(.99),100)
    y = [responseFun.eval(x) for x in X]
    ax.plot(X, y, **plot_kws)

def get_plot_info(subjid):
    plot_info = {}
    for dim in ['motion', 'orientation']:
        taskinfo, df = load_threshold_data(subjid, dim)
        if df is not None:
            # remove outliers
            subset = df.copy()
            """
            while True:
                outlier_max = subset.decision_var.mean()+subset.decision_var.std()*2.5
                outlier_min = subset.decision_var.mean()-subset.decision_var.std()*2.5
                filter_vec = (outlier_min<subset.decision_var) & \
                                (subset.decision_var<outlier_max)
                if filter_vec.mean() == 1:
                    break
                subset = subset[filter_vec]
            """
            nbins = min(11, len(subset)//10)
            # get accuracy as a function of binned decision variable
            # bins decision variables
            bins = subset.decision_var.quantile(np.linspace(0,1,nbins)); bins.iloc[-1]+=100
            subset.insert(0, 'decision_bins', np.digitize(subset.decision_var, bins))
            binned_decision_var = subset.groupby('decision_bins').decision_var.mean()
            subset.insert(0, 'binned_decision_var', subset.decision_bins.replace(binned_decision_var))
            bin_accuracy = subset.groupby('binned_decision_var').FB.agg(['mean', 'std', 'count'])
            # calculate standard error
            bin_accuracy.insert(0, 'se', bin_accuracy['std']/bin_accuracy['count']**.5)
            
            # get choice as a function of binned ending stimulus value
            if dim == 'motion':
                bins = subset.speed_change.quantile(np.linspace(0,1,nbins)); bins.iloc[-1]+=100
                subset.insert(0, 'stim_bins', np.digitize(subset.speed_change, bins))
                binned_stim_var = subset.groupby('stim_bins').speed_change.mean()
            else:
                bins = subset.ori_change.quantile(np.linspace(0,1,nbins)); bins.iloc[-1]+=100
                subset.insert(0, 'stim_bins', np.digitize(subset.ori_change, bins))
                binned_stim_var = subset.groupby('stim_bins').ori_change.mean()
            subset.insert(0, 'binned_stim_var', subset.stim_bins.replace(binned_stim_var))
            bin_response = subset.groupby('binned_stim_var').binarized_response.agg(['mean', 'std', 'count'])
            # calculate standard error
            bin_response.insert(0, 'se', bin_response['std']/bin_response['count']**.5)

            # update plot info
            plot_info[dim] = {'bin_accuracy': bin_accuracy,
                              'bin_response': bin_response,
                              'df': df}
    return plot_info
    
def plot_threshold_run(subjid, responseFun='lapseWeibull'):
    colors = ['m', 'c']
    plot_info = get_plot_info(subjid)
    sns.set_context('poster')
    f, axes = plt.subplots(3,2, figsize=(16,16))
    for i, key in enumerate(plot_info.keys()):
        # plot accuracy
        axes[0][i].errorbar(plot_info[key]['bin_accuracy'].index, 
                            plot_info[key]['bin_accuracy']['mean'], 
                            yerr=plot_info[key]['bin_accuracy']['se'].tolist(), 
                            marker='o',
                            linestyle="None",
                            c=colors[i])
        xlim = axes[0][i].get_xlim()
        # plot response fun fit
        estimate = .01 if key=='motion' else 8
        fitResponseCurve, metrics = fit_response_fun(plot_info[key]['df'].FB,
                                            plot_info[key]['df'].decision_var,
                                            estimate,
                                            kind=responseFun)
        plot_response_fun(fitResponseCurve, axes[0][i], plot_kws={'c': colors[i]})
        axes[0][i].set_ylabel('Accuracy', fontsize=24)
        axes[0][i].set_xlabel('Decision Var', fontsize=24)
        axes[0][i].set_title(key.title(), fontsize=30, y=1.05)
        axes[0][i].set_xlim(*xlim)
        # plot choice proportion
        axes[1][i].errorbar(plot_info[key]['bin_response'].index, 
                            plot_info[key]['bin_response']['mean'], 
                            yerr=plot_info[key]['bin_response']['se'].tolist(), 
                            marker='o',
                            linestyle="None",
                            c=colors[i])
        # plot fit logistic function
        stim_col = 'ori_change' if key=='orientation' else 'speed_change'
        fitClf, metrics = fit_choice_fun(plot_info[key]['df'], stim_col)
        plot_choice_fun(fitClf,
                        minval=plot_info[key]['df'].loc[:,stim_col].min(),
                        maxval=plot_info[key]['df'].loc[:,stim_col].max(),
                        ax=axes[1][i], 
                        plot_kws={'c': colors[i]})
        axes[1][i].set_ylabel('Positive Choice %', fontsize=24)
        axes[1][i].set_xlabel('%s Change' % key, fontsize=24)
        
        # plot quest estimate
        plot_info[key]['df'].quest_estimate.plot(ax=axes[2][i], c=colors[i])
        axes[2][i].set_ylabel('Quest Estimate', fontsize=24)
        axes[2][i].set_xlabel('Trial Number', fontsize=24)
        plt.subplots_adjust(hspace=.4)



