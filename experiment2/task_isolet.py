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
from rbm_tensorflow import *

from plotly.tools import FigureFactory as FF


trials = 50
kfold= 10
hidNeurons = 250
maxIterRbm = 15
 
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

#mydata= data(nul, dataset=None, percTrain=0.7, percVal=0.1, percTest=0.2, normType='max', shuf=True, posOut='last', outBin=True)

# print ( 'Loading train file...')
mydataset = np.genfromtxt(DATA_PATH+ 'isolet1+2+3+4.csv', delimiter=',')

acc1 = list()
tim1 = list()

acc2 = list()
tim2 = list()

acc3 = list()
tim3 = list()

normRBMELM = list()
normELM = list()
normELMRO = list()

for t in range(trials):
    trial= t +1
    print("TRIAL", trial)
    for j in range(kfold-1):
        i= j+1
        percTrain= round(0.1 * i, 1)
        percTest= 1 - percTrain  
        mydata= data(dataset=mydataset, train=None, val=None, test=None, 
                    percTrain=percTrain, percVal=0.0, percTest=percTest, 
                    normType='max', shuf=True, posOut='last', outBin=True)
        mydata.save("datasets/isolet")
        mydata.load()
        ###########################################################################
        print(line2())
        print (trial, 'Start training RBM ...')  
        init = time.time() # getting the start time
        rbmNet = RBM (dataIn=mydata.trainIn, numHid=400, rbmType='GBRBM')
        rbmNet.train (maxIter=maxIterRbm, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=150, freqPrint=10)
        W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)
        elmNet1 = ELM (neurons=hidNeurons,
                        inTrain=mydata.trainIn, 
                        outTrain=mydata.trainOut,
                        inputW=W,
                        beta=None,
                        batchSize=None)
        elmNet1.train(aval=True)
        end = time.time() # getting the end time
        res, a1 = elmNet1.getResult(mydata.testIn, mydata.testOut, True)
        nor,_ = elmNet1.getNorm()
        w1= elmNet1.getWeight()
        normRBMELM.append(nor)  
        acc1.append(a1)
        tim1.append(end-init)
        elmNet1.saveELM("log/isolet-rbm_" + str(trial) + "-" + str(percTrain) + "-" + str(percTest) + "_" + str(i))
        del(elmNet1)

        ###########################################################################
        print(line())
        print (trial, 'Start training ELM ...')  
        init2 = time.time()
        elmNet2 = ELM (neurons=hidNeurons,
                        inTrain=mydata.trainIn,
                        outTrain=mydata.trainOut,
                        inputW='uniform',
                        beta=None,
                        batchSize=None)
        elmNet2.train(aval=True)    
        end2 = time.time() # getting the end time     
        res, a2 = elmNet2.getResult(mydata.testIn,realOutput=mydata.testOut, aval=True)   
        nor,_ = elmNet2.getNorm()
        w2= elmNet2.getWeight()
        normELM.append(nor)
        acc2.append(a2)
        tim2.append(end2-init2)    
        elmNet2.saveELM("log/isolet-rnd_" + str(trial) + "-" + str(percTrain) + "-" + str(percTest) + "_" + str(i))
        del(elmNet2)
     
        ###########################################################################
        print(line())
        print (trial, 'Start training ELM-RO ...')  
        init3 = time.time()

        elmNet3 = ELM (neurons=hidNeurons,
                        inTrain=mydata.trainIn,
                        outTrain=mydata.trainOut,
                        inputW='RO',
                        beta=None,
                        batchSize=None)    
        elmNet3.train(aval=True)
        end3 = time.time()    
        res, a3 = elmNet3.getResult (mydata.testIn, realOutput=mydata.testOut,aval=True) 
        nor,_ = elmNet3.getNorm()
        normELMRO.append(nor)
        acc3.append(a3)
        tim3.append(end3-init3)
        elmNet3.saveELM("log/isolet-ro" + str(trial) + "-" + str(percTrain) + "-" + str(percTest) + "_" + str(i))
        del(elmNet3)
        
        
        ###########################################################################   
        # print ('\nIteration time: ', end3-init, ' sec', 'Predict to end: ', (end3-init)*(it-i)/60, ' min')
        
        # text= line2() + "ELM settings\nhidNeurons:" + str(hidNeurons)
        # text+= line() + "Weight RBM: " + br() + elmNet1.W
        # text+= line() + "Weight RND: " + br() + elmNet2.W
        # text+= line() + "Weight RO: " + br() + elmNet3.W
        # text+= line2()
        # log_iter("data", i, text)
      
        # del(rbmNet)
        # del(W)
        # del(elmNet1)
        # del(elmNet2)
        # del(elmNet3)
        gc.collect()
    
print ('######### RESULT ############' )

print ('RBM-ELM:')
acc1= np.asarray(acc1)
tim1 = np.asarray(tim1)
normRBMELM = np.asarray(normRBMELM)
# p(acc1)
print ('Accuracy - Mean: ', acc1.mean(), ' | Std: ', acc1.std())
print ('Time - Mean ', tim1.mean(), ' | Std: ', tim1.std())
print ('Norm - Mean ', normRBMELM.mean(), ' | Std: ', normRBMELM.std())
# text= acc1 + line() + tim1 + line() + normRBMELM
# log_uniq('elmrbm', text)

print ('\nOnly ELM:')
acc2 = np.asarray(acc2)
tim2 = np.asarray(tim2)
normELM = np.asarray(normELM)
# p(acc2)
print ('Accuracy -  mean: ', acc2.mean(), '| Std: ', acc2.std())
print ('Time - mean: ', tim2.mean(), ' | Std: ', tim2.std())
print ('Norm - Mean ', normELM.mean(), ' | Std: ', normELM.std())
# text= acc2 + line() + tim2 + line() + normELM
# log_uniq('elm', text)

print ('\nELM-RO:')
acc3 = np.asarray(acc3)
tim3 = np.asarray(tim3)
normELMRO = np.asarray(normELMRO)
# p(acc3)
print ('Accuracy -  mean: ', acc3.mean(), '| Std: ', acc3.std())
print ('Time - mean: ', tim3.mean(), ' | Std: ', tim3.std())
print ('Norm - Mean ', normELMRO.mean(), ' | Std: ', normELMRO.std())
# text= acc3 + line() + tim3 + line() + normELMRO
# log_uniq('elmro', text)


data = [acc1, acc2, acc3]
log_uniq("spambase", data)
plt.boxplot(data, labels=['RBM-ELM', 'ELM', 'ELM-RO'])

'''
statTest = statisticalTest (data, ['ELM', 'ELM-RO', 'RBM-ELM'], 0.05)
plt.saveResults (acc, acc2, acc3, statTest,['Acc RBM-ELM','Acc ELM','Acc ELM-RO','Statistical Test'], '/home/arissetyawan/APASCA/__THESIS___/ELM/codes/data')
plt.show()

'''



