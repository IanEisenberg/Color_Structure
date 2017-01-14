#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:44:06 2017

@author: ian
"""
import pandas as pd
import hddm
from hddm.utils import EZ
import numpy as np

gtrain_df = pd.read_pickle('Analysis_Output/gtrain_df.pkl')
gtrain_learn_df = pd.read_pickle('Analysis_Output/gtrain_learn_df.pkl')
gtest_learn_df = pd.read_pickle('Analysis_Output/gtest_learn_df.pkl')
gtest_conform_df = pd.read_pickle('Analysis_Output/gtest_conform_df.pkl')
gtest_df = pd.read_pickle('Analysis_Output/gtest_df.pkl')

gtrain_learn_df.id = gtrain_learn_df.id.astype('str').apply(lambda x: x.zfill(3))
gtest_learn_df.id = gtest_learn_df.id.astype('str').apply(lambda x: x.zfill(3))



df = gtest_learn_df.query('stim_conform == True')
for models in ['bias2','bias1','eoptimal', 'ignore', 'switch','memory','perseverance','permem']:
    df[models + '_certainty'] = (abs(df[models + '_posterior']-.5))/.5
 


ddm_df = df.loc[:,['rt','correct','id','context']]
ddm_df.columns = ['rt','response','subj_idx','context']

# use EZ, quick and dirty
subset = ddm_df.query('subj_idx == "028" and abs(context) < .5')
pc = subset['response'].mean()
vrt = np.var(subset.query('response == 1')['rt'])
mrt = np.mean(subset.query('response == 1')['rt'])
drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
print(drift,thresh,non_dec)


# Fit HDDM model
model = "v ~ context"
m = hddm.HDDMRegressor(ddm_df, model)
m.find_starting_values
m.sample(1000, burn=200, thin = 5)


