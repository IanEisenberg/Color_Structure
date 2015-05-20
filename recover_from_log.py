# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:03:22 2015

@author: Ian
"""

import numpy as np
import yaml

config_file = 'Config_Files/Prob_Context_034_config_2015-05-17_14-20-37.npy'
subject_code = '034'
datafilename = subject_code + '_Prob_Context_2015-05-17_14-20-37'

stimulusInfo = []
loaded_config = np.load(config_file)
for trial in loaded_config:
            if 'taskname' in trial.keys():
                taskinfo=trial
            else:
                stimulusInfo.append(trial)
timestamp = config_file[-23:-4]

#get data
f = open('Log/034_Prob_Context_2015-05-17_14-20-38.log')             
run1 = f.readlines()
run1 = run1[1:] #remove first taskinfo line
f = open('Log/034_second_run_Prob_Context_2015-05-17_15-09-14.log')             
run2 = f.readlines()
run2 = run2[1:]
run = run1 + run2


trials = []
for line in run:
    trials.append(yaml.load(line[0:-1]))

               

data = {}
data['taskinfo']=taskinfo
data['configfile']=config_file
data['subcode']=subject_code
data['timestamp']=timestamp
data['taskdata']=trials
f=open('RawData/' + datafilename + '.yaml','w')
yaml.dump(data,f)