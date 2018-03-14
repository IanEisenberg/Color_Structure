import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psychopy.data import FitCumNormal, FitWeibull
import seaborn as sns
from Dot_Task.Analysis.load_data import load_threshold_data
from Dot_Task.Exp_Design.utils import get_difficulties, get_trackers

def beautify_legend(legend, colors, fontsize=None):
    for i, text in enumerate(legend.get_texts()):
        text.set_color(colors[i])
    for item in legend.legendHandles:
        item.set_visible(False)
    legend.get_frame().set_linewidth(0.0)
    if fontsize:
        plt.setp(legend.get_texts(), fontsize=fontsize)

def fit_response_fun(df, kind='CumNorm'):
     df = df.query('exp_stage != "pause" and rt==rt')
     sigma = [1]*len(df)
     if kind == 'CumNorm':
         fun = FitCumNormal
     elif kind == 'FitWeibull':
         fun = FitWeibull
     return fun(df.decision_var, df.FB, sigma)

def plot_response_fun(responseFun, ax=None):
    if ax is None:
        f, ax = plt.subplots()
    X = np.linspace(0,responseFun.inverse(.99),100)
    y = [responseFun.eval(x) for x in X]
    ax.plot(X,y)

def get_plot_info(subj_code):
    plot_info = {}
    for dim in ['motion', 'orientation']:
        taskinfo, df = load_threshold_data(subj_code, dim)
        df = df.iloc[20:] # drop first trials where the threshold rapidly drops
        if df is not None:
            # get accuracy as a function of binned decision variable
            # bins decision variables
            bins = df.decision_var.quantile(np.linspace(0,1,11)); bins.iloc[-1]+=100
            df.insert(0, 'decision_bins', np.digitize(df.decision_var, bins))
            binned_decision_var = df.groupby('decision_bins').decision_var.mean()
            df.insert(0, 'binned_decision_var', df.decision_bins.replace(binned_decision_var))
            bin_accuracy = df.groupby('binned_decision_var').FB.agg(['mean', 'std', 'count'])
            # calculate standard error
            bin_accuracy.insert(0, 'se', bin_accuracy['std']/bin_accuracy['count']**.5)
            
            # get choice as a function of binned ending stimulus value
            if dim == 'motion':
                bins = df.speed_end.quantile(np.linspace(0,1,11)); bins.iloc[-1]+=100
                df.insert(0, 'stim_bins', np.digitize(df.speed_end, bins))
                binned_stim_var = df.groupby('stim_bins').speed_end.mean()
            else:
                bins = df.ori_end.quantile(np.linspace(0,1,11)); bins.iloc[-1]+=100
                df.insert(0, 'stim_bins', np.digitize(df.ori_end, bins))
                binned_stim_var = df.groupby('stim_bins').ori_end.mean()
            df.insert(0, 'binned_stim_var', df.stim_bins.replace(binned_stim_var))
            df.insert(0, 'binarized_response', df.response.replace({'up':1, 'down':0, 
                                                                    'right': 1, 'left': 0}))
            bin_response = df.groupby('binned_stim_var').binarized_response.agg(['mean', 'std', 'count'])
            # calculate standard error
            bin_response.insert(0, 'se', bin_response['std']/bin_response['count']**.5)

            # update plot info
            plot_info[dim] = {'bin_accuracy': bin_accuracy,
                              'bin_response': bin_response,
                              'df': df}
        return plot_info
    
def plot_threshold_run(subj_code):
    plot_info = get_plot_info(subj_code)
    sns.set_context('poster')
    f, axes = plt.subplots(3,2, figsize=(24,16))
    for i, key in enumerate(plot_info.keys()):
        # plot accuracy
        axes[0][i].errorbar(plot_info[key]['bin_accuracy'].index, 
                            plot_info[key]['bin_accuracy']['mean'], 
                            yerr=plot_info[key]['bin_accuracy']['se'].tolist(), 
                            marker='o')
        # plot fit cumNorm
        cumNorm = fit_response_fun(plot_info[key]['df'], kind='CumNorm')
        plot_response_fun(cumNorm, axes[0][i])
        axes[0][i].set_ylabel('Accuracy', fontsize=24)
        axes[0][i].set_xlabel('Decision Var', fontsize=24)
        axes[0][i].set_title(key.title(), fontsize=30, y=1.05)
        # plot choice
        axes[1][i].errorbar(plot_info[key]['bin_response'].index, 
                            plot_info[key]['bin_response']['mean'], 
                            yerr=plot_info[key]['bin_response']['se'].tolist(), 
                            marker='o')
        axes[1][i].set_ylabel('Positive Choice %', fontsize=24)
        axes[1][i].set_xlabel('%s Ending Value' % key, fontsize=24)
        # plot quest estimate
        plot_info[key]['df'].groupby('speedStrength')\
            .quest_estimate.plot(ax=axes[2][i], legend=True)
        axes[2][i].set_ylabel('Quest Estimate', fontsize=24)
        axes[2][i].set_xlabel('Trial Number', fontsize=24)
        leg = axes[2][i].get_legend()
        colors = [l.get_color() for l in axes[2][i].get_lines()]
        beautify_legend(leg, colors)
        plt.subplots_adjust(hspace=.4)



