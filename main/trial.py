#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:18:24 2023

@author: aswinpaul
"""

from trial_clmethod import trial

# call the trail function with memory horizon 
# data will be saved to folder: data_n_plot
memory_horizon = 5

# trial function executes function from file: trial_clmethod
trial(memory_horizon)