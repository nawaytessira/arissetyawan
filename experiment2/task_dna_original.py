# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

"""

MAIN_DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/'
DATA_PATH= MAIN_DIR + 'datasets/'

import sys
sys.path.insert (0, MAIN_DIR)
sys.path.insert (0, '/usr/local/lib/python3.6/site-packages/')

import tensorflow as tf
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
import scipy
import plotly.plotly as py

from elm_ori import *
from utilsClassification import *
from dataManipulation import data
from rbm_tensorflow import *

from plotly.tools import FigureFactory as FF



it = 30
hidNeurons = 250
maxIterRbm = 15

# loading the data set
print ('Loading the dataset...')

dnaTest = np.genfromtxt(DATA_PATH + 'dna_test.csv', delimiter=',')
dnaVal = np.genfromtxt(DATA_PATH + 'dna_val.csv', delimiter=',')
dnaTrain = np.genfromtxt(DATA_PATH + 'dna_train.csv', delimiter=',')
dnaTrain = np.concatenate((dnaTrain,dnaVal))

acc = list()
tim = list()

acc2 = list()
tim2 = list()

acc3 = list()
tim3 = list()

dna = data (train=dnaTrain, test=dnaTest, val=dnaVal, posOut='first')

normRBMELM = list()
normELM = list()
normELMRO = list()


for i in range(it):

    
    ###########################################################################
    print ('Starting training RBM ', i , ' ...'  )
    
    init = time.time() # getting the start time
    rbmNet = RBM_TF (dataIn=dna.trainIn, numHid=hidNeurons, rbmType='GBRBM')
    #rbmNet.train (maxIter=maxIterRbm, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=150, freqPrint=10)
    rbmNet.train (maxIter=maxIterRbm, lr=0.01, wc=0.01, iMom=0.5, fMom=0.9, cdIter=1, batchSize=250, freqPrint=10)
    W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)        
    del(rbmNet)    
    
    print ('Starting training RBM-ELM ', i , ' ...' )
    elmNet = ELM_ORI (hidNeurons, dna.trainIn, dna.trainOut, W)
    elmNet.train(aval=True)    
    end = time.time() # getting the end time     
    res, a = elmNet.getResult (dna.testIn,dna.testOut,True)
    
    nor,_ = elmNet.getNorm()
    normRBMELM.append(nor)  
    acc.append(a)
    tim.append(end-init)
    del(elmNet)
    
    ###########################################################################
    print( '\n\n')
    print( 'Starting training ELM ', i , ' ...' )
    init2 = time.time()
    elmNet = ELM_ORI (hidNeurons, dna.trainIn, dna.trainOut)
    elmNet.train(aval=True)    
    end2 = time.time() # getting the end time     
    res, a = elmNet.getResult (dna.testIn,dna.testOut,True)   
    
    nor,_ = elmNet.getNorm()
    normELM.append(nor)
    acc2.append(a)
    tim2.append(end2-init2)    
    del(elmNet)

    ###########################################################################
    print ('\n\n')
    print ('Starting training ELM-RO ', i , ' ...' )
    init3 = time.time()    
    elmNet = ELM_ORI (hidNeurons, dna.trainIn, dna.trainOut, init='RO')
    elmNet.train(aval=True)     
    end3 = time.time()    
    res, a = elmNet.getResult (dna.testIn,dna.testOut,True) 
    
    nor,_ = elmNet.getNorm()
    normELMRO.append(nor)
    acc3.append(a)
    tim3.append(end3-init3)
    del(elmNet)
    
    ###########################################################################   
    print ('\nIteration time: ', end3-init, ' sec', 'Predict to end: ', (end3-init)*(it-i)/60, ' min')
    
    #del(dna)   
    gc.collect()
	
print ('######### DNA ',maxIterRbm, ' ############' )
print('Both RBM-ELM:')
acc = np.asarray(acc)
tim = np.asarray(tim)
normRBMELM = np.asarray(normRBMELM)
print('Accuracy - Mean: ', acc.mean(), ' | Std: ', acc.std())
print('Time - Mean ', tim.mean(), ' | Std: ', tim.std())
print('Norm - Mean ', normRBMELM.mean(), ' | Std: ', normRBMELM.std())

print('\nOnly ELM:')
acc2 = np.asarray(acc2)
tim2 = np.asarray(tim2)
normELM = np.asarray(normELM)
print('Accuracy -  mean: ', acc2.mean(), '| Std: ', acc2.std())
print('Time - mean: ', tim2.mean(), ' | Std: ', tim2.std())
print('Norm - Mean ', normELM.mean(), ' | Std: ', normELM.std())

print('\nOnly ELM-RO:')
acc3 = np.asarray(acc3)
tim3 = np.asarray(tim3)
normELMRO = np.asarray(normELMRO)
print('Accuracy -  mean: ', acc3.mean(), '| Std: ', acc3.std())
print('Time - mean: ', tim3.mean(), ' | Std: ', tim3.std())
print('Norm - Mean ', normELMRO.mean(), ' | Std: ', normELMRO.std())

#data = [acc2, acc3, acc]
#boxplot(data, nPlots=1,  posSubPlot=[1,1], names=['ELM', 'ELM-RO', 'RBM-ELM'], xLabel='Algorithms', yLabel='Accuracy(%)', saveName='/home/patcha/ResultadosRO/dna', figSize=[17,7])
##boxplot(data, nPlots=1,  posSubPlot=[1,1], names=['ELM', 'ELM-RO', 'RBM-ELM'], xLabel='Algorithms', yLabel='Accuracy(%)', figSize=[17,7])
#statTest = statisticalTest (data, ['ELM', 'ELM-RO', 'RBM-ELM'], 0.05)
#
#saveResults (acc, acc2, acc3, statTest,['Acc RBM-ELM','Acc ELM','Acc ELM-RO','Statistical Test'], '/home/patcha/ResultadosRO/dna')










