# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

Revised: Aris Setyawan
E-mail: arissetyawan.email@gmail.com

"""

MAIN_DIR= "/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/"
DATA_PATH= MAIN_DIR + "datasets/"

import sys
sys.path.insert(0, MAIN_DIR)
sys.path.insert(0, "/usr/local/lib/python3.6/site-packages/")

import numpy as np
import time
import gc
import matplotlib.pyplot as plt
import scipy
import plotly.plotly as py
import pandas as pd

from elm import *
from utilsClassification import *
from dataManipulation import *
from rbm_tensorflow import *
from sklearn.preprocessing import LabelEncoder
from plotly.tools import FigureFactory as FF

dataset='australian'
fold_number= 1
K= 10
neuron=None
if len(sys.argv)!= 2:
    print("Please pass argument for neuron number eg. 100")
else:
    neuron= int(sys.argv[1])
    print("Neurons: ", neuron)
    maxIterRbm = 200
    print("Loading the dataset...")
    objFold= Fold(dataset, K, 'last')
    objFold.Stratified()

    # loading the data set
    # The __init__ parameters:    
    # dataset: the whole dataset. Default = None. You need to upload the split datasets with load method
    # percTrain: the % of train data
    # percVal: the % of validation data
    # percTest: the % of test data
    # Shuf: If you wanna shuffle the dataset set it as True, otherwise, False
    # posOut: The output position in the dataset. You can choose: last, for the last column
    # or first, for the first column. If there is no output, set it as None.    
    # outBin: if it"s true, the output will rise one bit for each position. Ex: if the output
    # is 3, the binary output will be an array [0, 0. 1].
    # If the dataset has already been splitted, you can upload all the partitions using
    # train, val and test. 


    acc1 = list()
    tim1 = list()

    acc2 = list()
    tim2 = list()

    acc3 = list()
    tim3 = list()

    normRBMELM = list()
    normELM = list()
    normELMRO = list()
 
    for fold in range(K):
        train= objFold.loadTrain(fold_number)
        test= objFold.loadTest(fold_number)
        mydata= data(train=train, test=test, val=None, normType="maxmin", shuf=False, posOut="last", outBin=True)
        mydata.save("datasets/"+dataset+"_iterfold_"+ str(fold_number))

        ###########################################################################
        print(line2())
        print (fold_number, "Start training ELM-RBM ...")
        init = time.time() # getting the start time
        rbmNet = RBM (dataIn=mydata.trainIn, numHid=neuron, rbmType="GBRBM")
        # lr = 0.001, wc = 0.01, iMom,fMom = 0.5/0.9 and bs = 100 @hinton, 2010
        rbmNet.train (maxIter=maxIterRbm, lr=0.001, wc=0.01, iMom=0.5, fMom=0.9, batchSize=100)
        W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)
        elmNet1 = ELM (neurons=neuron,
                        inTrain=mydata.trainIn, 
                        outTrain=mydata.trainOut,
                        inputW=W,
                        dataName=dataset)
        elmNet1.train(aval=True)
        end = time.time() # getting the end time
        res, a1 = elmNet1.getResult(mydata.testIn, mydata.testOut, True)
        nor,_ = elmNet1.getNorm()
        w1= elmNet1.getWeight()
        normRBMELM.append(nor)  
        acc1.append(a1)
        tim1.append(end-init)
        elmNet1.saveELM("log/"+ dataset + "/" + str(neuron)  + "rbm_" + str(fold_number))
        del(elmNet1)

        ###########################################################################
        print(line())
        print (fold_number, "Start training ELM-RND...")  
        init2 = time.time()
        elmNet2 = ELM (neurons=neuron,
                        inTrain=mydata.trainIn,
                        outTrain=mydata.trainOut,
                        inputW="uniform",
                        dataName=dataset)
        elmNet2.train(aval=True)
        end2 = time.time() # getting the end time     
        res, a2 = elmNet2.getResult(mydata.testIn,realOutput=mydata.testOut, aval=True)   
        nor,_ = elmNet2.getNorm()
        w2= elmNet2.getWeight()
        normELM.append(nor)
        acc2.append(a2)
        tim2.append(end2-init2)
        elmNet2.saveELM("log/"+ dataset  + "/" + str(neuron)  + "rnd_" + str(fold_number))
        del(elmNet2)

        ###########################################################################
        print(line())
        print (fold_number, "Start training ELM-RO ...")  
        init3 = time.time()

        elmNet3 = ELM (neurons=neuron,
                        inTrain=mydata.trainIn,
                        outTrain=mydata.trainOut,
                        inputW="RO",
                        dataName=dataset)
        elmNet3.train(aval=True)
        end3 = time.time()
        res, a3 = elmNet3.getResult (mydata.testIn, realOutput=mydata.testOut,aval=True) 
        nor,_ = elmNet3.getNorm()
        normELMRO.append(nor)
        acc3.append(a3)
        tim3.append(end3-init3)
        elmNet3.saveELM("log/"+ dataset + "/" + str(neuron) + "ro_" + str(fold_number))
        del(elmNet3)

        fold_number += 1
        gc.collect()
            
    text= "\n\n######### RESULT ############"
    print (text)
    acc1= np.asarray(acc1)
    tim1 = np.asarray(tim1)
    normRBMELM = np.asarray(normRBMELM)
    print ("\nRBM-ELM::")
    log("log/" + dataset + "/"+str(neuron)+ "elmrbm-acc", acc1)
    log("log/" + dataset + "/"+str(neuron)+ "elmrbm-tim", tim1)
    log("log/" + dataset + "/"+str(neuron)+ "elmrbm-norm", normRBMELM)
    log("log/" + dataset + "/"+str(neuron)+ "elmrbm-std", str(acc1.std()))
    text += "\nRBM-ELM:"
    text += "\nAccuracy - Mean: " + str(acc1.mean()) + " | Std: "  + str(acc1.std())
    text += "\nTime - Mean "  + str(tim1.mean()) + " | Std: " + str(tim1.std())
    text += "\nNorm - Mean " + str(normRBMELM.mean()) + " | Std: "  + str(normRBMELM.std())

    print ("\nOnly ELM:")
    acc2 = np.asarray(acc2)
    tim2 = np.asarray(tim2)
    normELM = np.asarray(normELM)
    log("log/" + dataset + "/"+str(neuron)+ "elmrnd-acc", acc2)
    log("log/" + dataset + "/"+str(neuron)+ "elmrnd-tim", tim2)
    log("log/" + dataset + "/"+str(neuron)+ "elmrnd-norm", normELM)
    log("log/" + dataset + "/"+str(neuron)+ "elmrnd-std", str(acc2.std()))
    text += "\nELM-RND:"
    text += "\nAccuracy - Mean: " + str(acc2.mean()) + " | Std: " + str(acc2.std())
    text += "\nTime - Mean "  + str(tim2.mean()) + " | Std: " + str(tim2.std())
    text += "\nNorm - Mean " + str(normELM.mean()) + " | Std: "  + str(normELM.std())

    print ("\nELM-RO:")
    acc3 = np.asarray(acc3)
    tim3 = np.asarray(tim3)
    normELMRO = np.asarray(normELMRO)
    log("log/" + dataset + "/"+str(neuron)+ "elmro-acc", acc3)
    log("log/" + dataset + "/"+str(neuron)+ "elmro-tim", tim3)
    log("log/" + dataset + "/"+str(neuron)+ "elmro-norm", normELM)
    log("log/" + dataset + "/"+str(neuron)+ "elmro-std", str(acc3.std()))
    text += "\nELM-RO:"
    text += "\nAccuracy - Mean: " + str(acc3.mean()) + " | Std: "  + str(acc3.std())
    text += "\nTime - Mean "  + str(tim3.mean()) + " | Std: " + str(tim3.std())
    text += "\nNorm - Mean " + str(normELMRO.mean()) + " | Std: "  + str(normELMRO.std())

    print ("\nALL RESULT:")
    data = [acc1, acc2, acc3]
    log("log/" + dataset + "/"+str(neuron)+ "acc-all", data)
    log("log/" + dataset + "/"+str(neuron)+ "result", text)

    # plt.boxplot(data, labels=["RBM-ELM", "ELM", "ELM-RO"])