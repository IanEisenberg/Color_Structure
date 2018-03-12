import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Dot_Task.Analysis.load_data import load_threshold_data
from Dot_Task.Exp_Design.utils import get_difficulties, get_trackers

def plot_weibull(alpha, beta=3.5, chance = .5):
    x = np.linspace(0,alpha*3,100)
    y = chance + (1.0-chance)*(1-np.exp( -(x/alpha)**(beta) ))
    plt.plot(x,y)
    return alpha

def plot_threshold_run(subj_code, dim='motion'):
    taskinfo, df = load_threshold_data(subj_code, dim)
    difficulties = get_difficulties(subj_code)[dim]

    # bins decision variables
    bins = df.decision_var.quantile(np.linspace(0,1,11))
    df.insert(0, 'binned_decision_var', np.digitize(df.decision_var, bins))
    bin_accuracy = df.groupby('binned_decision_var').FB.agg(['mean', 'std', 'count'])
    # calculate standard error
    bin_accuracy.insert(0, 'se', bin_accuracy['std']/bin_accuracy['count']**.5)
    # combine with bins
    bin_accuracy.index = bins.index
    bin_accuracy = pd.concat([bins, bin_accuracy], axis=1).iloc[:-1]
    
    sns.set_context('poster')
    f = plt.figure(figsize=(12,8))
    plt.errorbar(bin_accuracy['decision_var'], bin_accuracy['mean'], 
                 yerr=bin_accuracy['se'].tolist(), marker='o')
    plt.ylabel('Accuracy', fontsize=24)
    plt.xlabel('Decision Var', fontsize=24)
