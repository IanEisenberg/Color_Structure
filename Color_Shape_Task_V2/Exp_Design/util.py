#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:58:26 2017

@author: nantanick
"""

import numpy as np

def setBeta(seed = None):
    """
    Sets a random beta distribution from a pre-defined catalog
    ------------------------------
    seed -- set seed (default None)
    ------------------------------
    """
    if seed is not None:
        np.random.seed(seed)
    beta_catalog = [[20, 21],[2, 4],[3, 23],[1, 15]]
    rand_num = np.random.randint(0,len(beta_catalog)-1)
    beta_ab = beta_catalog[rand_num]
    np.random.shuffle(beta_ab)
    # center & scale the beta distribution from 0-1 to -1 - 1 
    loc = -1
    scale = 2
    return [{'a':beta_ab[0], 'b':beta_ab[1], 'loc':loc, 'scale':scale},
            {'a':beta_ab[1], 'b':beta_ab[0], 'loc':loc, 'scale':scale}]
                 