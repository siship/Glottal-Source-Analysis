#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 02:40:58 2018

@author: sishir
"""


from scipy.io.wavfile import read as wavread
#from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy
import numpy as np
import librosa
from removeTrend import removeTrend
#samplerate, x = wavfile.read('/home/sishir/My_Home/FilesIITG/CourseWork/EE_628_speech/assignment_3/a_30ms.wav')

#[samplerate, sig] = wavread('')
   # print '\nReading with scipy.io.wavfile.read: ', x 
#sig = scipy.signal.resample(sig, 8000)
    
sig, samplerate = librosa.load('/home/sishir/My_Home/sishir_nasals/ForTesting/mha_angami.wav', sr=8000) # Downsample 44.1kHz to 8kHzmsec = samplerate/1000
msec = samplerate/1000

sig = [ele/max(abs(sig)) for ele in sig]
#sig = abs(scipy.signal.hilbert(sig1))

lenSig =  len(sig)

windowSizeForTrendRemove = int(8*msec)
window = scipy.signal.boxcar(windowSizeForTrendRemove)

sigDiff = np.diff(sig)
sigDiff = np.ndarray.tolist(sigDiff)
lastItem = sigDiff[lenSig-2]
sigDiff.append(lastItem)
sigDiffA = np.asarray(sigDiff)

#bCoeff = np.asarray(1, dtype='float')
#aCoeff = np.asarray([1, -4, 6, -4, 1],  dtype='float')
#zi = scipy.signal.lfilter_zi(bCoeff, aCoeff)
#xx = scipy.signal.lfilter(bCoeff, aCoeff, sigDiffA, zi=zi*sigDiffA[0])
zfSig = np.cumsum(np.cumsum(np.cumsum(np.cumsum(np.cumsum(np.cumsum(sigDiffA))))));

zfSig = removeTrend(zfSig,  window, windowSizeForTrendRemove)
zfSig = removeTrend(zfSig,  window, windowSizeForTrendRemove)
zfSig = removeTrend(zfSig,  window, windowSizeForTrendRemove)
zfSig = removeTrend(zfSig,  window, windowSizeForTrendRemove)

zfSig[int(lenSig-windowSizeForTrendRemove*2):lenSig]=0;
zfSig[1:windowSizeForTrendRemove*2]=0;

# For GCI detection
ZC = np.diff((zfSig>0)*1)
pnZC = np.where(ZC == 1)[0]
npZC = np.where(ZC == -1)[0]

pES = np.abs(zfSig[pnZC + 1] - zfSig[pnZC - 1])
nES = np.abs(zfSig[npZC + 1] - zfSig[npZC - 1])

avgpes = np.mean(pES)
avgnes = np.mean(nES)

if avgpes>avgnes:
        gci = pnZC
        es = pES
else:
        gci = npZC
        es = nES 



#plt.subplot(2, 1, 1)
#plt.plot(sig, 'b')
#plt.subplot(2, 1, 2)
#plt.stem(es)
#plt.show()


plt.plot(sig, 'b', zorder=1)
plt.stem(gci, np.ones(len(gci)), zorder=2)
plt.show()

