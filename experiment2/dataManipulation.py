# -*- coding: utf-8 -*-
"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

Improved: Aris Setyawan
E-mail: arissetyawan.email@gmail.com

This class just implements a data class to ease the dataset manipulation.
If you find some bug, plese e-mail me =)

"""


MAIN_DIR= "/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/"
DATA_PATH= MAIN_DIR + "datasets/"

import sys
import pandas as pd
import numpy as np
sys.path.insert (0, MAIN_DIR)
sys.path.insert (0, "/usr/local/lib/python3.6/site-packages/")

class Fold:
    K = None
    dataset= None
    train_file= None
    fold_file= None
    test_file= None

    def __init__(self, dataset, K):
        print("Fold initialising...", dataset, K)
        self.df= pd.read_csv(DATA_PATH+ dataset +".csv", header=None)
        self.dataset=  dataset
        self.K= K

    def remove_tab(self, file):
        f = open(file, "r")
        out= f.read().replace(" ","")
        # print(out)
        f.close
        f = open(file, "w")
        f.write(out)
        f.close

    def split(self):
        if K < 2:
            print("Please pass argument for split at least: 2")
        else:
            print("Splitting into n folds:  ", K)
            folds= np.array_split(df, K)
            i = 1
            for fold in folds:
                print(i)
                self.fold_file= DATA_PATH+ self.dataset + "_fold_" + str(i) + ".csv"
                np.savetxt(self.fold_file, fold, delimiter=",", fmt='%10.2f')
                self.remove_tab(self.fold_file)
                i += 1

    def loadFold(self, fold_number):
      return np.genfromtxt( DATA_PATH+ self.dataset + "_" + str(self.K) + "fold_" + str(fold_number) + ".csv", delimiter=',')

    def loadTest(self, fold_number):
      return np.genfromtxt( DATA_PATH+ self.dataset + "_" + str(self.K) + "foldtest_" + str(fold_number) + ".csv", delimiter=',')

    def loadTrain(self, fold_number):
      return np.genfromtxt( DATA_PATH+ self.dataset + "_" + str(self.K) + "foldtrain_" + str(fold_number) + ".csv", delimiter=',')

    def KFold(self):
        result= []
        folds= np.array_split(self.df, self.K)
        for i in range(self.K):
            fold_number= i + 1
            # print(' ------------- Fold: ', fold_number)
            fold= folds[i]
            test= fold
            train= []
            for j in range(self.K):
                if i!= j:
                    train.append(folds[j])
            train= np.concatenate(train)
            # print("TEST", test)
            # print("TRAIN", train)
            result.append([test, train])
            self.fold_file= DATA_PATH+ self.dataset + "_" + str(self.K) + "fold_" + str(fold_number) + ".csv"
            self.train_file= DATA_PATH+ self.dataset + "_" + str(self.K) + "foldtrain_" + str(fold_number) + ".csv"
            self.test_file= DATA_PATH+ self.dataset + "_" + str(self.K) + "foldtest_" + str(fold_number) + ".csv"

            np.savetxt(self.fold_file, fold, delimiter=",", fmt='%10.2f')
            np.savetxt(self.train_file, train, delimiter=",", fmt='%10.2f')
            np.savetxt(self.test_file, test, delimiter=",", fmt='%10.2f')

            self.remove_tab(self.fold_file)
            self.remove_tab(self.train_file)
            self.remove_tab(self.test_file)
            i += 1
        return result


import numpy as np
from utilsClassification import *
# The data must be a numpy array in the format [# of samples, # of attributes]. The output
# can be stored either in the first or last column. If there is no output, set posOut = None
class data:
    trainIn = None      # The input train data
    trainOut = None     # The output train data
    valIn = None        # The input validation data
    valOut = None       # The output validation data
    testIn = None       # The input test data
    testOut = None      # The output test data
    normType = 'max'    # The normalization type. You can choose one of them:
                        # max: normalize by the max
                        # mean: normalize with mean=0 and std=1
                        # maxmin: normalize by column using the max and min
                        # None: no normalize
    nSamples = None     # The number of samples
    nClass = None       # The number of classes in the dataset
    
    
    
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
    def __init__(self, dataset=None, train=None, val=None, test=None, percTrain=0.7, percVal=0.1, percTest=0.2, normType='max', shuf=True, posOut='last', outBin=True):
        if (percTrain+percVal+percTest) != 1.0:
            raise Exception('Initialize error: the percTrain+percVal+percTest must be 1')
        if posOut != 'last' and posOut != 'first' and posOut is not None:
            raise Exception('Initialize error: check your posOut type')
        if normType != 'max' and normType != 'mean' and normType != 'maxmin' and normType != 'maxminmean' and normType is not None:
            raise Exception('Initialize error: check your normType')        
        
        self.normType = normType
        
        # So, the dataset is not splitted
        if dataset is not None:
            p("Using whole dataset")  
            if normType is not None:
                self.normalize (dataset, normType, posOut)            
            else:
               p("normType is None")  

            # print (dataset)
            
            # Shuffling
            if shuf:
               p("Shuffling............")
               dataset = np.random.permutation(dataset)             
            else:
               p("not shuffling")
            # Getting the number of samples
            self.nSamples, nAtt = dataset.shape
            
            # Getting the number of samples for each partition
            nSTrain = int(round(self.nSamples*percTrain))
            nSVal = int(round(self.nSamples*percVal))
            nSTest = self.nSamples - nSTrain - nSVal
            # Splitting
            self.split (dataset, posOut, nSTrain, nSVal, nSTest)
            
            # Checking if the outpu will be binary
            if outBin == True and posOut is not None:
                self.binOut ()
            else:
                p("adjustInt")
                self.adjustInt()
                p("reshapeOutput")
                self.reshapeOutput()
        else:
            # So, in this case the load method must be used 
            if train is None and val is None and test is None:
                self.normType = None
            
            else:
                # Shuffling
                if shuf:
                    p("shuffling......")
                    if train is not None:
                        train = np.random.permutation(train)
                    if val is not None:
                        val = np.random.permutation(val)
                    if test is not None:
                        test = np.random.permutation(test)
                # Splitting                                     
                self.splitOuts (train, val, test, posOut)
                
                self.normalizeSplited (normType, posOut)
                
                # Binarizing
                if outBin == True and posOut is not None:
                    self.binOut ()
                else:
                    p("reshapeOutput")
                    self.reshapeOutput()
            
            self.nSamples = 0
            if train is not None:
                self.nSamples += self.trainIn.shape[0] 
            if test is not None:
                self.nSamples += self.testIn.shape[0]
            if val is not None:
                self.nSamples += self.valIn.shape[0]
        
        self.nClass = self.trainOut.shape[1]
        
        # noise
        #self.testIn = self.testIn + (0.2 * self.testIn.std() * np.random.random(self.testIn.shape))
                
        
    def __str__(self):
        out = '### Data Class ###'
        out = out + '\nNumber of samples = (' + str(self.nSamples) + ')'
        out = out + '\nNumber of classes = (' + str(self.nClass) + ')'
        if self.trainIn is not None:            
            out = out + '\ntrainIn = (' + str(self.trainIn.shape) + ')'
        else:
            out = out + '\ntrainIn = (None)'
        if self.trainOut is not None:
            out = out + '\ntrainOut = (' + str(self.trainOut.shape) + ')'
        else:
            out = out + '\ntrainOut = (None)'
        if self.valIn is not None:
            out = out + '\nvalIn = (' + str(self.valIn.shape) + ')'
        else:
            out = out + '\nvalIn = (None)'
        if self.valOut is not None:
            out = out + '\nvalOut = (' + str(self.valOut.shape) + ')'
        else:
            out = out + '\nvalOut = (None)'
        if self.testIn is not None:       
            out = out + '\ntestIn = (' + str(self.testIn.shape) + ')'
        else:
            out = out + '\ntestIn = (None)'
        if self.testOut is not None:
            out = out + '\ntestOut = (' + str(self.testOut.shape) + ')'
        else:
            out = out + '\ntestOut = (None)'        
        if self.normType is not None:
            out = out + '\nNormalization: ' + self.normType        
        else:
            out = out + '\nNormalization: None'        
        return out
    
    def normalize (self, dataset, normType, posOut):
        p("Normalizing the dataset...", normType)
        if posOut is None:
            if normType == None:
                dataset= dataset
            elif normType == 'max':
                dataset = dataset/dataset.max()
            elif normType == 'mean':
                dataset = (dataset - dataset.mean())/dataset.std()
            elif normType == 'maxmin' or normType == 'maxminmean':
                dataset = self.normMaxMinByColumn(dataset, normType)
        elif posOut == 'first':
            if normType == None:
                dataset= dataset
            elif normType == 'max':
                dataset[:,1:] = dataset[:,1:]/dataset[:,1:].max()
            elif normType == 'mean':
                dataset[:,1:] = (dataset[:,1:] - dataset[:,1:].mean())/dataset[:,1:].std()
            elif normType == 'maxmin' or normType == 'maxminmean':
                dataset[:,1:] = self.normMaxMinByColumn(dataset[:,1:], normType)
        elif posOut == 'last':
            if normType == None:
                dataset= dataset
            elif normType == 'max':
                dataset[:,0:-1] = dataset[:,0:-1]/dataset[:,0:-1].max()
            elif normType == 'mean':
                dataset[:,0:-1] = (dataset[:,0:-1] - dataset[:,0:-1].mean())/dataset[:,0:-1].std()          
            elif normType == 'maxmin' or normType == 'maxminmean':
                dataset[:,0:-1] = self.normMaxMinByColumn(dataset[:,0:-1], normType)
                
    def normalizeSplited (self, normType, posOut):
        if posOut is None:
            if normType == 'max':
                if self.trainIn is not None:
                    self.trainIn = self.trainIn/self.trainIn.max()
                if self.testIn is not None:
                    self.testIn = self.testIn/self.testIn.max()
                if self.valIn is not None:
                    self.valIn = self.valIn/self.valIn.max()
                
            elif normType == 'mean':
                if self.trainIn is not None:
                    self.trainIn = (self.trainIn - self.trainIn.mean())/self.trainIn.std()                    
                if self.testIn is not None:
                    self.testIn = (self.testIn - self.testIn.mean())/self.testIn.std()                    
                if self.valIn is not None:
                    self.valIn = (self.valIn - self.valIn.mean())/self.valIn.std()                 
                
            elif normType == 'maxmin' or normType == 'maxminmean':
                if self.trainIn is not None:                     
                    self.trainIn = self.normMaxMinByColumn(self.trainIn, normType)
                if self.testIn is not None:
                    self.testIn = self.normMaxMinByColumn(self.testIn, normType)                    
                if self.valIn is not None:
                    self.valIn = self.normMaxMinByColumn(self.valIn, normType)               
                
        elif posOut == 'first':
            if normType == 'max':
                if self.trainIn is not None:
                    self.trainIn[:,1:] = self.trainIn[:,1:]/self.trainIn[:,1:].max()
                if self.testIn is not None:
                    self.testIn[:,1:] = self.testIn[:,1:]/self.testIn[:,1:].max()
                if self.valIn is not None:
                    self.valIn[:,1:] = self.valIn[:,1:]/self.valIn[:,1:].max()
 
            elif normType == 'mean':
                if self.trainIn is not None:
                    self.trainIn[:,1:]  = (self.trainIn[:,1:]  - self.trainIn[:,1:].mean())/self.trainIn[:,1:].std()                    
                if self.testIn is not None:
                    self.testIn[:,1:] = (self.testIn[:,1:] - self.testIn[:,1:].mean())/self.testIn[:,1:].std()                    
                if self.valIn is not None:
                    self.valIn[:,1:] = (self.valIn[:,1:] - self.valIn[:,1:].mean())/self.valIn[:,1:].std()                    
                
            elif normType == 'maxmin' or normType == 'maxminmean':
                if self.trainIn is not None:                     
                    self.trainIn[:,1:]  = self.normMaxMinByColumn(self.trainIn[:,1:] , normType)
                if self.testIn is not None:
                    self.testIn[:,1:] = self.normMaxMinByColumn(self.testIn[:,1:] , normType)                    
                if self.valIn is not None:
                    self.valIn[:,1:]  = self.normMaxMinByColumn(self.valIn[:,1:] , normType)                

        elif posOut == 'last':
            if normType == 'max':
                if self.trainIn is not None:
                    self.trainIn[:,0:-1] = self.trainIn[:,0:-1]/self.trainIn[:,0:-1].max()
                if self.testIn is not None:
                    self.testIn[:,0:-1] = self.testIn[:,0:-1]/self.testIn[:,0:-1].max()
                if self.valIn is not None:
                    self.valIn[:,0:-1] = self.valIn[:,0:-1]/self.valIn[:,0:-1].max()
 
            elif normType == 'mean':
                if self.trainIn is not None:
                    self.trainIn[:,0:-1] = (self.trainIn[:,0:-1]  - self.trainIn[:,0:-1].mean())/self.trainIn[:,0:-1].std()                    
                if self.testIn is not None:
                    self.testIn[:,0:-1] = (self.testIn[:,0:-1] - self.testIn[:,0:-1].mean())/self.testIn[:,0:-1].std()                    
                if self.valIn is not None:
                    self.valIn[:,0:-1] = (self.valIn[:,0:-1] - self.valIn[:,0:-1].mean())/self.valIn[:,0:-1].std()                    
                
            elif normType == 'maxmin' or normType == 'maxminmean':
                if self.trainIn is not None:                     
                    self.trainIn[:,0:-1] = self.normMaxMinByColumn(self.trainIn[:,0:-1] , normType)
                if self.testIn is not None:
                    self.testIn[:,0:-1] = self.normMaxMinByColumn(self.testIn[:,0:-1] , normType)                    
                if self.valIn is not None:
                    self.valIn[:,0:-1] = self.normMaxMinByColumn(self.valIn[:,0:-1] , normType)            

        
    def normMaxMinByColumn (self, data, normType):
        mi = data.min(axis=0)
        ma = data.max(axis=0)
        m,_ = data.shape
                
        for i in range(m):
            data[i,:] = (data[i,:]-mi)/(ma-mi)
            
        if normType == 'maxminmean':
            data = (data - data.mean())/data.std()
        
        return data
    
    def normMaxByColumn (self, data):
        ma = data.max(axis=0)
        _,n = data.shape
                
        for i in range(n):
            data[i,:] = data[i,:]/ma
        
        return data    
            
    def split (self, dataset, posOut, nSTrain, nSVal, nSTest):
        p("Splitting the dataset...posOut, nSTrain, nSVal, nSTest", posOut, nSTrain, nSVal, nSTest)
        if posOut is None:  
            if nSTrain > 0:
                self.trainIn = dataset[0:nSTrain,:]
            if nSVal > 0:
                self.valIn = dataset[nSTrain:nSTrain+nSVal,:]
            if nSTest > 0:
                self.testIn = dataset[nSTrain+nSVal:self.nSamples,:]            
        elif posOut == 'first':
            if nSTrain > 0:
                self.trainIn = dataset[0:nSTrain,1:]
                self.trainOut = dataset[0:nSTrain,0]
            if nSVal > 0:
                self.valIn = dataset[nSTrain:nSTrain+nSVal,1:]
                self.valOut = dataset[nSTrain:nSTrain+nSVal,0]
            if nSTest > 0:
                self.testIn = dataset[nSTrain+nSVal:self.nSamples,1:] 
                self.testOut = dataset[nSTrain+nSVal:self.nSamples,0] 
        elif posOut == 'last':
            if nSTrain > 0:
                self.trainIn = dataset[0:nSTrain,0:-1]
                p(dataset)       
                self.trainOut = dataset[0:nSTrain,-1]
            if nSVal > 0:
                self.valIn = dataset[nSTrain:nSTrain+nSVal,0:-1]
                self.valOut = dataset[nSTrain:nSTrain+nSVal,-1]
            if nSTest > 0:
                self.testIn = dataset[nSTrain+nSVal:self.nSamples,0:-1] 
                self.testOut = dataset[nSTrain+nSVal:self.nSamples,-1]
        p(posOut)
        p("trainIn", self.trainIn)
        p("trainOut", self.trainOut)
        p("valIn", self.valIn)
        p("valOut", self.valOut)
        p("testIn", self.testIn)
        p("testOut", self.testOut)

    def splitOuts (self, train, val, test, posOut):
        if posOut is None:  
            if train is not None:
                self.trainIn = train
            if val is not None:
                self.valIn = val
            if test is not None:
                self.testIn = test
        elif posOut == 'first':
            if train is not None:
                self.trainIn = train[:,1:]
                self.trainOut = train[:,0]
            if val is not None:
                self.valIn = val[:,1:]
                self.valOut = val[:,0]
            if test is not None:
                self.testIn = test[:,1:] 
                self.testOut = test[:,0] 
        elif posOut == 'last':
            if train is not None:
                self.trainIn = train[:,0:-1]
                self.trainOut = train[:,-1]
            if val is not None:
                self.valIn = val[:,0:-1]
                self.valOut = val[:,-1]
            if test is not None:
                self.testIn = test[:,0:-1] 
                self.testOut = test[:,-1]        

    def binOut (self):
        p("binOut")
        if self.trainOut is not None:
            self.trainOut[self.trainOut < 0] = 0
            minClass = self.trainOut.min()
            p(self.trainOut)
            self.trainOut = ind2vec(self.trainOut - minClass)
            if self.valOut is not None:
                self.valOut[self.valOut < 0] = 0
                self.valOut = ind2vec(self.valOut - minClass)
            if self.testOut is not None:
                self.testOut[self.testOut < 0] = 0
                self.testOut = ind2vec(self.testOut - minClass)

    def adjustInt (self):
        minClass = self.trainOut.min()  
        if minClass == 0:            
            self.trainOut = self.trainOut+1
            self.testOut = self.testOut+1
    
    def reshapeOutput (self):
        if self.trainOut is not None:
            self.trainOut = np.reshape(self.trainOut,(self.trainOut.shape[0],1)) 
            
        if self.testOut is not None:
            self.testOut = np.reshape(self.testOut,(self.testOut.shape[0],1))
            
        if self.valOut is not None:
            self.valOut = np.reshape(self.valOut,(self.valOut.shape[0],1))
        
    
    # Saving all partitions
    def save (self,name='data', ext='.csv'):
        p("Saving all partition of datasets...")
        if self.trainIn is not None:     
            np.savetxt(name+'-trainIn'+ext,self.trainIn, fmt='%10.2f')

        if self.trainOut is not None:
            np.savetxt(name+'-trainOut'+ext,self.trainOut, fmt='%10.2f')

        if self.valIn is not None:
            np.savetxt(name+'-valIn'+ext,self.valIn, fmt='%10.2f')

        if self.valOut is not None:
            np.savetxt(name+'-valOut'+ext,self.valOut, fmt='%10.2f')

        if self.testIn is not None:       
            np.savetxt(name+'-testIn'+ext,self.testIn, fmt='%10.2f')

        if self.testOut is not None:
            np.savetxt(name+'-testOut'+ext,self.testOut, fmt='%10.2f')
    
    # loading the partitions saved previously with the method save
    def load (self, trainIn=None, trainOut=None, valIn=None, valOut=None, testIn=None, testOut=None):
        p("call load all the dataset...")
        if trainIn is not None:                        
            self.trainIn = trainIn

        if trainOut is not None:
            self.trainOut = trainOut
        if valIn is not None:
            self.valIn = valIn
            
        if valOut is not None:
            self.valOut = valOut

        if testIn is not None:       
            self.testIn = testIn

        if testOut is not None:
            self.testOut = testOut
    

    
              
