# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

Revised: Aris Setyawan
E-mail: arissetyawan.email@gmail.com

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

from elm import *
from utilsClassification import *
from dataManipulation import data
from rbm import *
#from rbm_tensorflow import *

from plotly.tools import FigureFactory as FF


it = 100
hidNeurons = 300
maxIterRbm = 100
 
# loading the data set
print('Loading the dataset...')
    # The __init__ parameters:    
    # dataset: the whole dataset. Default = None. You need to upload the split datasets with load method
    # percTrain: the % of train data
    # percVal: the % of validation data
    # percTest: the % of test data
    # Shuf: If you wanna shuffle the dataset set it as True, otherwise, False
    # posOut: The output position in the dataset. You can choose: last, for the last column
    # or first, for the first column. If there is no output, set it as None.    
    # outBin: if it's true, the output will rise one bit for each position. Ex: if the output
    # is 3, the binary output will be an array [0, 0. 1].
    # If the dataset has already been splitted, you can upload all the partitions using
    # train, val and test. 

#     def __init__(self, dataset=None, train=None, val=None, test=None, percTrain=0.7, percVal=0.1, percTest=0.2, normType='max', shuf=True, posOut='last', outBin=True):

#spambase= data(nul, dataset=None, percTrain=0.7, percVal=0.1, percTest=0.2, normType='max', shuf=True, posOut='last', outBin=True)

print ( 'Loading train file...')
spambaseTrain = np.genfromtxt(DATA_PATH+ 'spambase.data', delimiter=',')

acc = list()
tim = list()

acc2 = list()
tim2 = list()

acc3 = list()
tim3 = list()

spambase = data (dataset=spambaseTrain, posOut='last')
print(spambase)

normRBMELM = list()
normELM = list()
normELMRO = list()


for i in range(it):

    
    ###########################################################################
    print ('*' * i)
    print ( 'Starting training RBM ', i , ' ...')  
    
    init = time.time() # getting the start time
    rbmNet = RBM (dataIn=spambase.trainIn, numHid=400, rbmType='GBRBM')
    #rbmNet.train (maxIter=maxIterRbm, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=150, freqPrint=10)
    rbmNet.train (maxIter=maxIterRbm, lr=0.01, wc=0.01, iMom=0.5, fMom=0.9, cdIter=1, batchSize=150, freqPrint=10)
    W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)        
    del(rbmNet)    
    
    elmNet = ELM (hidNeurons, spambase.trainIn, spambase.trainOut, W)
    elmNet.train(aval=False)    
    end = time.time() # getting the end time     
    res, a = elmNet.getResult (spambase.testIn,spambase.testOut,True)
    
    nor,_ = elmNet.getNorm()
    normRBMELM.append(nor)  
    acc.append(a)
    tim.append(end-init)    
    del(elmNet)
    
    ###########################################################################
    print ('\n\n')
    print ('Starting training ELM ', i , ' ...' )
    init2 = time.time()
    elmNet = ELM (hidNeurons, spambase.trainIn, spambase.trainOut)
    elmNet.train(aval=False)    
    end2 = time.time() # getting the end time     
    res, a = elmNet.getResult (spambase.testIn,spambase.testOut,True)   
    
    nor,_ = elmNet.getNorm()
    normELM.append(nor)
    acc2.append(a)
    tim2.append(end2-init2)    
    del(elmNet)

    ###########################################################################
    print ('\n\n')
    print ('Starting training ELM-RO ', i , ' ...' )
    init3 = time.time()
    elmNet = ELM (hidNeurons, spambase.trainIn, spambase.trainOut, init='RO')
    elmNet.train(aval=False)
    end3 = time.time()    
    res, a = elmNet.getResult (spambase.testIn,spambase.testOut,True) 
    
    nor,_ = elmNet.getNorm()
    normELMRO.append(nor)
    acc3.append(a)
    tim3.append(end3-init3)
    del(elmNet)
    
    ###########################################################################   
    print ('\nIteration time: ', end3-init, ' sec', 'Predict to end: ', (end3-init)*(it-i)/60, ' min')
    
    #del(spambase)   
    gc.collect()
	
print ('######### spambase ############' )
print ('\nOnly ELM:')
acc2 = np.asarray(acc2)
tim2 = np.asarray(tim2)
normELM = np.asarray(normELM)
print ('Accuracy -  mean: ', acc2.mean(), '| Std: ', acc2.std())
print ('Time - mean: ', tim2.mean(), ' | Std: ', tim2.std())
print ('Norm - Mean ', normELM.mean(), ' | Std: ', normELM.std())

print ('\nOnly ELM-RO:')
acc3 = np.asarray(acc3)
tim3 = np.asarray(tim3)
normELMRO = np.asarray(normELMRO)
print ('Accuracy -  mean: ', acc3.mean(), '| Std: ', acc3.std())
print ('Time - mean: ', tim3.mean(), ' | Std: ', tim3.std())
print ('Norm - Mean ', normELMRO.mean(), ' | Std: ', normELMRO.std())

print ('Both RBM-ELM:')
acc = np.asarray(acc)
tim = np.asarray(tim)
normRBMELM = np.asarray(normRBMELM)
print ('Accuracy - Mean: ', acc.mean(), ' | Std: ', acc.std())
print ('Time - Mean ', tim.mean(), ' | Std: ', tim.std())
print ('Norm - Mean ', normRBMELM.mean(), ' | Std: ', normRBMELM.std())


data = [acc, acc2, acc3]
plt.boxplot(data, labels=['RBM-ELM', 'ELM', 'ELM-RO'])

'''
statTest = statisticalTest (data, ['ELM', 'ELM-RO', 'RBM-ELM'], 0.05)
plt.saveResults (acc, acc2, acc3, statTest,['Acc RBM-ELM','Acc ELM','Acc ELM-RO','Statistical Test'], '/home/arissetyawan/APASCA/__THESIS___/ELM/codes/spambase')
plt.show()

'''



