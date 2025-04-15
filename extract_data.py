# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 04:52:03 2018

@author: fakher
"""
import _extract_data

def extract(name):
    rate, sampwidth, data = _extract_data.readwav(name)
    return rate, sampwidth, data
