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
sys.path.insert (0, MAIN_DIR)
sys.path.insert (0, "/usr/local/lib/python3.6/site-packages/")

import tensorflow as tf
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

dataset=''
fold_number= 1
K= 4
neurons=[100,200,300,400,500]

if len(sys.argv)!= 3:
    print("Please pass argument for dataset name and label position eg. iris last")
else:
    dataset= sys.argv[1]
    posOut= sys.argv[2]
    print("Loading the dataset with label position: ", dataset, posOut)
    if dataset=="gissete":
        data= pd.read_csv(DATA_PATH+ dataset +".csv", header=None)
        labels= pd.read_csv(DATA_PATH+ dataset +".labels", header=None)
        objFold= Stratified(dataset, K, data, labels)
        objFold.load()
    else:
        objFold= Fold(dataset, K, posOut)
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

    for neuron in neurons: 

        acc2 = list()
        tim2 = list()
        normELM = list()

        print("Neuron:", neuron)

        for fold in range(K):
            fold_number= fold + 1
            train= objFold.loadTrain(fold_number)
            test= objFold.loadTest(fold_number)
            # print(test)
            mydata= data(train=train, test=test, val=None, normType="maxmin", shuf=False, posOut=posOut, outBin=True)
            mydata.save("datasets/"+dataset+ str(neuron) + "_iterfold_"+ str(fold_number))
            mydata.load()

        #     ###########################################################################
            print(line2())
           
        #     ###########################################################################
            print (neuron, fold_number, "Start training ELM-RND...")  
            init2 = time.time()
            elmNet2 = ELM (neurons=neuron,
                            inTrain=mydata.trainIn,
                            outTrain=mydata.trainOut,
                            inputW="uniform",
                            dataName=dataset)
            elmNet2.train(aval=True)
            end2 = time.time() # getting the end time 
            res, a2 = elmNet2.getResult(mydata.testIn, realOutput=mydata.testOut, aval=True)   
            nor,_ = elmNet2.getNorm()
            w2= elmNet2.getWeight()
            acc2.append(a2)
            normELM.append(nor)
            tim2.append(end2-init2)
            elmNet2.saveELM("log/"+ dataset  + "/" + str(neuron)  + "rnd_" + str(fold_number))
            del(elmNet2)

            gc.collect()
                
        text= "\n\n######### RESULT ############"
        print (text)

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

        print ("\nALL RESULT:")
        log("log/" + dataset + "/"+str(neuron)+ "acc-all", acc2)
        log("log/" + dataset + "/"+str(neuron)+ "result", text)
