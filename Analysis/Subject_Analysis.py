# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:12:43 2015

@author: Ian
"""


import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Load_Data import load_data
from ggplot import *

name = 'Pilot002_Temp_Struct_2015-02-24_16-00-20'
data_file = '../Data/' + name + '.yaml'
taskinfo, df, dfa = load_data(data_file, name)

#*********************************************
# Set up plotting defaults
#*********************************************

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18,
        }
        
axes = {'titleweight' : 'bold'
        }
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('figure', figsize = (6,6))

#*********************************************
# Set up helper functions
#*********************************************
def window_function(func, array, window):
    slices = len(array)/window
    windowed_scores = [float(func(array[i*window:(i+1)*window])) for i in range(slices)]
    if slices*window < len(array):
        windowed_scores.append(float(func(array[(slices)*window:])))
    return np.array(windowed_scores)

def bar(x, y, title):
    plot = plt.bar(x,y)
    plt.title(str(title))
    return plot
    

#*********************************************
# Preliminary Analysis
#*********************************************

block_len = taskinfo['block_len']


#Basic things - look at distribution of RT, etc.
plt.plot(dfa.rt*1000,'ro')
plt.title('RT over experiment', size = 24)
plt.xlabel('trial')
plt.ylabel('RT in ms')



# If subjects have learned a taskset, rather than individual actions, then subjects
# should switch to the other task set after an error (rather than simply explore)
# We calculate the proportion of time the subject maintains the same taskset after
# a win or a loss. 

window = 30
win_stay = [dfa.switch[i] == False and dfa.FB.shift(1)[i] == 1 for i in dfa.index]
lose_stay = [dfa.switch[i] == False and dfa.FB.shift(1)[i] == 0 for i in dfa.index]

wstay_sum = window_function(sum,win_stay,window)
lstay_sum = window_function(sum,lose_stay,window)
win_sum = window_function(sum, dfa.FB.shift(1) == 1, window)
lose_sum = window_function(sum, dfa.FB.shift(1) == 0, window)
wstay_mean = wstay_sum/win_sum
lstay_mean = lstay_sum/lose_sum
stay_mean = window_function(np.mean,[not dfa.switch[i] for i in dfa.index], window)

#Another way to look at taskset learning is using 
# P(TS1 key at t | TS1 key at T-1)/P(TS1)
# Just the P(stay) for each TS individually.
TS1stay=sum([not dfa.switch[i] for i in dfa.index if dfa.curr_ts[i] == 0])
TS2stay=sum([not dfa.switch[i] for i in dfa.index if dfa.curr_ts[i] == 1])



#We can plot these probabilities binned in temporal blocks over time:
samples = np.arange(0,len(dfa),window)
fig = plt.figure(figsize = (8,8))
plt.plot(samples,wstay_mean, samples, lstay_mean, samples, stay_mean, 'r--')
plt.xlabel('Trial #')
plt.ylabel('Probability of continuing taskset')
plt.legend(['P(Stay|Win)', 'P(Stay|Lose)', 'P(Stay)'], loc = 'best')

# Probability of X based on position in inferred block
# P(switch|lose) for each position in the inferred block

starts = [0, len(dfa.index)/2, 0]
ends = [len(dfa.index)/2, len(dfa.index), len(dfa.index)]

for start, end in zip(starts,ends):
    block = 12
    
    lswitch_position = np.array([np.mean([dfa.switch[i] == True for i in dfa.index[start:end] 
                            if dfa.FB.shift(1)[i]==0 and (i-1)%block == j]) 
                            for j in range(block)])
    # P(switch|win) for each position in the block                           
    wswitch_position = np.array([np.mean([dfa.switch[i] == True for i in dfa.index[start:end] 
                            if dfa.FB.shift(1)[i]==1 and (i-1)%block == j]) 
                            for j in range(block)])
    # P(switch for each position in the block                                                 
    switch_position = np.array([np.mean([dfa.switch[i] == True for i in dfa.index[start:end] 
                            if (i-1)%block == j]) for j in range(block)])
    # P(lose) for each position in block
    FB_position = np.array([np.mean([dfa.FB[i] for i in dfa.index[start:end] 
                            if (i-1)%block == j]) for j in range(block)])
    
    # RT for each position in block
    RT_position = np.array([np.mean([dfa.rt[i] for i in dfa.index[start:end] 
                            if (i-1)%block == j]) for j in range(block)])
                                
    
    plt.figure(figsize = (8,8))
    plt.suptitle('Stats in trial range: (' + str(start) + ' - ' + str(end) +')', fontsize = 28, y = 1.05)
    plt.subplot(221)
    bar(range(block),FB_position,'Average FB')
    plt.subplot(222)
    bar(range(block),switch_position,'P(switch)')
    plt.subplot(223)
    bar(range(block),lswitch_position,'P(switch | lose)')
    plt.xlabel('trial number mod ' + str(block))
    plt.subplot(224)
    bar(range(block),wswitch_position,'P(switch | win)')
    plt.xlabel('trial number mod ' + str(block))
    plt.tight_layout()


#Average distance between switches
a=[i  for i in dfa.index if dfa.switch[i] == True]
avg_distance = [(x,x - a[i-1]) for i,x in enumerate(a)][1:]
error_index = [i for i in dfa.index if dfa.FB[i] == 0]

plt.figure(figsize=(8,8))
plt.plot([x[0] for x in avg_distance], [x[1] for x in avg_distance],error_index, [1]*len(error_index),'|')
plt.axhline(block, color = 'red', linestyle = '--')
plt.title('Time between switches')
plt.xlabel('trial numer')
plt.ylabel('Trials since last switch')
plt.legend(['Switch Distance','Lose Event','Block Length'], loc = 'best')


# ggplot test
#ggplot(dfa.reset_index(), aes(x = 'trial_count', y = 'rt')) + geom_point() \
#     + stat_smooth(method = 'lm') + facet_wrap('switch')


#***************************
#Test analysis
#***************************

name = 'Pilot003_Temp_Struct_noFBTest_2015-02-24_15-39-52'
data_file = '../Data/' + name + '.yaml'
taskinfo, df, dfa = load_data(data_file, name, mode = 'test')
a=[i  for i in dfa.index if dfa.switch[i] == True]
avg_distance = [(x,x - a[i-1]) for i,x in enumerate(a)][1:]

plt.figure(figsize=(8,8))
plt.plot([x[0] for x in avg_distance], [x[1] for x in avg_distance])
plt.axhline(block, color = 'red', linestyle = '--')
plt.title('Time between switches')
plt.xlabel('trial numer')
plt.ylabel('Trials since last switch')
plt.legend(['Switch Distance','Lose Event','Block Length'], loc = 'best')


