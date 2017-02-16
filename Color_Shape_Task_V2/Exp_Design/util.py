#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:58:26 2017

@author: nantanick
"""

import numpy as np

def setBeta():
    beta_catalog = [[20, 21],[2, 4],[3, 23],[1, 15]]
    rand_num = np.random.randint(0,len(beta_catalog)-1)
    beta_ab = beta_catalog[rand_num]
    np.random.shuffle(beta_ab)
    return [{'a':beta_ab[0], 'b':beta_ab[1]},{'a':beta_ab[1], 'b':beta_ab[0]}]
                 