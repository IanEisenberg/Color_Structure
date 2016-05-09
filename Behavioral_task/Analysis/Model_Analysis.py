# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:41:57 2016

@author: ian
"""

from helper_functions import simulateModel
from helper_classes import BiasPredModel
from helper_functions import fit_bias2_model, fit_bias1_model, fit_static_model
import random as r
 
import pandas as pd
from scipy.stats import norm
import seaborn as sns

ts_dis = [norm(-.3,.37), norm(.3,.37)]
exp_len = 800
p = .9
num_sims = 100

off = []
for _ in range(num_sims):
    r1 = r.random()*.5
    r2 = r.random()*.5
    model = BiasPredModel(ts_dis, [.5,.5], r1, r2)
    df = simulateModel(model,ts_dis,'bias')
    bias2 = fit_bias2_model(ts_dis, df, action_eps = 0, model_type = 'TS', return_out = False)
    off.append((r1-bias2['r1'])**2 + (r2-bias2['r2'])**2)
