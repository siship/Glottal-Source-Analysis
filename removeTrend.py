#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 01:23:58 2018

@author: sishir
"""
import numpy
import numpy as np
def removeTrend(zfSig,  window, windowSizeForTrendRemove):
   
    rm = numpy.convolve(zfSig, window);

    rm = rm[int(np.ceil(windowSizeForTrendRemove/2)-1):int(len(rm)-(windowSizeForTrendRemove-np.ceil(windowSizeForTrendRemove/2)))];

    norm = numpy.convolve(np.ones(len(zfSig)),window)
    norm =norm[int(np.ceil(windowSizeForTrendRemove/2)-1):int(len(norm)-(windowSizeForTrendRemove-np.ceil(windowSizeForTrendRemove/2)))];
    rm = numpy.divide(rm, norm) 
    out = zfSig-rm;

    return out