import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Analysis.load_data import load_threshold_data
from Analysis.utils import fit_choice_fun, fit_response_fun

def beautify_legend(legend, colors=None, fontsize=None):
    if colors is None:
        colors = [i.get_color() for i in legend.legendHandles]
    for i, text in enumerate(legend.get_texts()):
        text.set_color(colors[i])
    for item in legend.legendHandles:
        item.set_visible(False)
    for item in legend.legendHandles:
        item.set_visible(False)
    if fontsize:
        plt.setp(legend.get_texts(), fontsize=fontsize)

def plot_choice_fun(clf, minval, maxval, ax=None, plot_kws={}):
    if ax is None:
        f, ax = plt.subplots()
    X = np.linspace(minval, maxval, 100)
    y = [i[1] for i in clf.predict_proba(X.reshape(-1,1))]
    ax.plot(X, y, **plot_kws)
    
def plot_response_fun(responseFun, ax=None, plot_kws=None):
    xlim = None
    if plot_kws is None:
        plot_kws = {}
    max_acc = responseFun.eval(np.inf)
    maxX = responseFun.inverse(max_acc-.001)
    minX = responseFun.inverse(.51)
    if ax is None:
        f, ax = plt.subplots()
    else:
        xlim = ax.get_xlim()
        maxX = xlim[1]        
    X = np.linspace(minX,maxX,100)
    y = [responseFun.eval(x) for x in X]
    ax.plot(X, y, **plot_kws)
    # plot points of interest
    y_points = [.7, .775, .85]
    x_points = responseFun.inverse(y_points)
    ax.plot(x_points, y_points, 'o', color='blue',
            markeredgecolor='white', markeredgewidth=1, markersize=9,
            zorder=10)
    if xlim:
        maxX = max(xlim[1], X[-1])
        ax.set_xlim(minX, maxX)

    
    
def get_plot_info(subjid, N=None):
    plot_info = {}
    for dim in ['motion', 'orientation']:
        taskinfo, df, files = load_threshold_data(subjid, dim)
        if df is not None:
            if N: df = df.iloc[-N:].query('rt==rt')
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
            nbins = min(7, len(subset)//10)
            # get accuracy as a function of binned decision variable
            # bins decision variables
            bins = subset.decision_var.quantile(np.linspace(0,1,nbins)); bins.iloc[-1]+=100
            subset.insert(0, 'decision_bins', np.digitize(subset.decision_var, bins))
            binned_decision_var = subset.groupby('decision_bins').decision_var.mean()
            subset.insert(0, 'binned_decision_var', subset.decision_bins.replace(binned_decision_var))
            bin_accuracy = subset.groupby('binned_decision_var').correct.agg(['mean', 'std', 'count'])
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
    
def plot_threshold_run(subjid, responseFun='lapseWeibull', N=None, size=8):
    colors = ['m', 'c']
    plot_info = get_plot_info(subjid, N=N)
    sns.set_context('paper',font_scale=1)
    f, axes = plt.subplots(3,2, figsize=(size*.7,size))
    for i, key in enumerate(plot_info.keys()):
        # plot accuracy
        axes[0][i].errorbar(plot_info[key]['bin_accuracy'].index, 
                            plot_info[key]['bin_accuracy']['mean'], 
                            yerr=plot_info[key]['bin_accuracy']['se'].tolist(), 
                            marker='o',
                            markersize=7,
                            markeredgecolor='white', 
                            markeredgewidth=1,
                            linestyle="None",
                            c=colors[i],)
        # plot response fun fit
        init_estimate = .01 if key=='motion' else 6
        fitResponseCurve, metrics = fit_response_fun(plot_info[key]['df'].correct,
                                            plot_info[key]['df'].decision_var,
                                            init_estimate,
                                            kind=responseFun)
        plot_response_fun(fitResponseCurve, axes[0][i], plot_kws={'c': colors[i]})
        axes[0][i].set_ylabel('Accuracy', fontsize=size*1.5)
        axes[0][i].set_xlabel('Decision Var', fontsize=size*1.5)
        axes[0][i].set_title(key.title(), y=1.05, fontsize=size*2)
        # plot choice proportion
        axes[1][i].errorbar(plot_info[key]['bin_response'].index, 
                            plot_info[key]['bin_response']['mean'], 
                            yerr=plot_info[key]['bin_response']['se'].tolist(), 
                            marker='o',
                            markersize=7,
                            markeredgecolor='white', 
                            markeredgewidth=1,
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
        axes[1][i].set_ylabel('Positive Choice %', fontsize=size*1.5)
        axes[1][i].set_xlabel('%s Change' % key, fontsize=size*1.5)
        
        # plot quest estimate
        plot_info[key]['df'].quest_estimate.plot(ax=axes[2][i], c=colors[i], )
        plot_info[key]['df'].decision_var.plot(ax=axes[2][i], c=colors[i], linestyle='--')
        axes[2][i].set_ylabel('Quest Estimate', fontsize=size*1.5)
        axes[2][i].set_xlabel('Trial Number', fontsize=size*1.5)
        plt.subplots_adjust(hspace=.4)
    return f

def plot_threshold_estimates(subjid, responseFun='lapseWeibull', 
                             window=300, step=5, size=8):
    colors = ['m', 'r', 'c']
    plot_info = get_plot_info(subjid)
    sns.set_context('paper',font_scale=1)
    f, axes = plt.subplots(1,2, figsize=(size,size*.7))
    for i, key in enumerate(plot_info.keys()):
        init_estimate = .01 if key=='motion' else 6
        FB = plot_info[key]['df'].correct
        decision_var = plot_info[key]['df'].decision_var
        win_difficulties = []
        # calculate fits in different windows
        win_range = range(0, max(len(FB)-window,1), step)
        for j in win_range:
            FB_win = FB.iloc[j:window+j]
            decision_var_win = decision_var.iloc[j:window+j]
            fitResponseCurve, metrics = fit_response_fun(FB_win,
                                                         decision_var_win,
                                                         init_estimate,
                                                         kind=responseFun)
            win_difficulties.append(fitResponseCurve.inverse([.7,.775, .85]))
        # plot the fits over windows
        zipped = list(zip(*win_difficulties))
        axes[i].plot(win_range, zipped[0], 'o-',
                    color=colors[0])
        axes[i].plot(win_range, zipped[1], 'o-',
                    color=colors[1])
        axes[i].plot(win_range, zipped[2], 'o-',
                    color=colors[2])
        axes[i].set_xlabel('Starting Trial Number', fontsize=size*1.5)
        axes[i].set_ylabel('Estimated Value', fontsize=size*1.5)
        axes[i].set_title(key, fontsize=size*2)
        # add overal estimate
        fitResponseCurve, metrics = fit_response_fun(FB,
                                                     decision_var,
                                                     init_estimate,
                                                     kind=responseFun)
        difficulties = fitResponseCurve.inverse([.7,.775,.85])
        axes[i].hlines(difficulties[0], 0, axes[i].get_xlim()[1],
                        linestyle='--', color=colors[0])
        axes[i].hlines(difficulties[1], 0, axes[i].get_xlim()[1],
                        linestyle='--', color=colors[1])
        axes[i].hlines(difficulties[2], 0, axes[i].get_xlim()[1],
                        linestyle='--', color=colors[2])
        
        
    leg = axes[0].legend(['85% accuracy', '77.5% accuracy', '70% accuracy'], 
              fontsize=size*1.5, markerscale=0)
    beautify_legend(leg)
    plt.suptitle('Windowed Estimates Window Size: %s' % window, fontsize=size*1.5)
