#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:31:26 2017

@author: ian
"""
import numpy as np
import os
import pandas as pd
import sys
sys.path.append('Exp_Design')
sys.path.append('Analysis')
from utils import get_trackers, plot_weibull, fit_weibull
from Load_Data import load_data, load_threshold_data
from matplotlib import pyplot as plt
from psychopy.data import FitWeibull, QuestHandler 

subj_code = 'IE2'
ts,df = load_threshold_data(subj_code)
trackers = get_trackers(subj_code)


# final tracker estimates
df.groupby(['speedStrength','motionDirection']).quest_estimate.agg(lambda x: x.iloc[-1]).plot(kind='bar')

# look at motion tackers
motion_trackers = trackers[0]
index=1
for key,tracker in motion_trackers.items():
    plt.subplot(len(motion_trackers)/2,2,index)
    alpha = fit_weibull(tracker.intensities,tracker.data).params[0]
    plot_weibull(alpha)
    plt.title(key)
    index+=1
plt.tight_layout()
    
tracker_df = pd.DataFrame()
for key,tracker in motion_trackers.items():
    df = pd.DataFrame({'difficulty': key[1],
                       'pedestal': key[0],
                       'data': tracker.data,
                       'intensities': tracker.intensities})
    tracker_df = pd.concat([tracker_df,df], axis=0)

# bin intensities
bins =np.linspace(0,tracker_df.intensities.max()/2,
                                                          10)
tracker_df.loc[:,'binned_intensities'] = bins[np.digitize(tracker_df.intensities,
                                              bins)-1]

# fit weibull
df = tracker_df.query('difficulty == "hard" and pedestal=="in"')
plot_weibull(df.intensities,df.data)
ax = plt.gca()
df.groupby('binned_intensities').data.mean().plot(style='bo', markersize=10, ax=ax)

fit = FitWeibull(df.intensities,df.data)


alpha = plot_weibull(intensities,data)
