# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This class implements the Extreme Learning Machine (ELM) according to:

[1] Huang, G.B.; Zhu, Q.Y.; Siew, C.-K. Extreme learning machine: theory and applications.
Neurocomputing, v. 70, n. 1, p. 489 - 501, 2006.

You can choose initialize the input weights with uniform distribution or using random orthogonal projection
proposed by:

[2] Wenhui W. and Xueyi L. ; The selection of input weights of extreme learning machine: A sample
structure preserving point of view, Neurocomputing, 2017, in press

Using this class you can either train the net or just execute if you already know the ELM's weight.
All the code is very commented to ease the undertanding.

Revised: Aris Setyawan
E-mail: arissetyawan.email@gmail.com

If you find some bug, please e-mail me =)

'''

import numpy as np
import sys

# Insert the paths
MAIN_DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/__THESIS___/experiment2/'
sys.path.insert (0, MAIN_DIR)

from utilsClassification import *
# from utilsClassification import sigmoid, cont_error
from sklearn.decomposition import PCA
from rbm import RBM


class ELM:
    neurons = None
    inTrain = None
    outTrain = None
    W = None
    beta = None
    P = None
    batchSize = None
    inputW= None
    dataName= None
    # The constructor method. If you intend to train de ELM, you must fill all parameters.
    # If you already have the weights and wanna only execute the net, just fill W and beta.
    def __init__ (self, neurons=20, inTrain=None, outTrain=None, inputW='uniform', beta=None, batchSize=None, dataName=None):
        # print("Initialize parameters:", neurons, inTrain, outTrain, W, beta, init, batchSize)
        # Setting the neuron's number on the hidden layer        
        self.neurons = neurons
        # Here we add 1 into the input's matrices to vectorize the bias computation
        self.inTrain = np.concatenate ((inTrain, np.ones([inTrain.shape[0],1])), axis = 1)
        self.outTrain = outTrain
        self.inputW= inputW
        self.dataName= dataName
        p(self.inTrain)
        p(self.outTrain)
        if inTrain is not None and outTrain is not None:          
            # If you wanna initialize the weights W, you just set it up as a parameter. If don't,
            # let the W=None and the code will initialize it here with random values
            p("inTrain OK, outTrain OK")
            if str(inputW) == 'uniform' or inputW is None:
                p("uniform !!!!!!!!!!!!!!!!!")
                self.W = np.random.uniform(-1,1,[inTrain.shape[1]+1,neurons])             
            elif str(inputW) == 'RO':            
                p("RO !!!!!!!!!!!!!!!!!")        
                if neurons >= inTrain.shape[1]:
                    self.W = np.random.uniform(-1,1,[inTrain.shape[1]+1,neurons])
                    self.W,_ = np.linalg.qr(self.W.T)
                    self.W = self.W.T
                else:   
                    print('Starting PCA...')
                    A = np.random.uniform(-1,1,[neurons,neurons])
                    A,_ = np.linalg.qr(A.T)
                    A = A.T
                    pca = PCA(n_components=neurons)                    
                    wpca = pca.fit_transform(inTrain.T).T

                    self.W = np.dot(A,wpca).T                        
                    # including the bias
                    b = np.random.uniform(-1,1,[1,self.W.shape[1]])                        
                    self.W = np.vstack((self.W,b))   
            else:
                p("RRRRRRRRRRRBM !!!!!!!!!!!!!!!!!")        
                self.W = inputW

        else:
            p("inTrain NO, outTrain NO")
            # In this case, there is no traning. So, you just to fill the weights W and beta
            if beta is not None and inputW is not None:
                self.beta = beta
                self.W = inputW
            else:
                print('ERROR: you set up the input training as None, but you did no initialize the weights')
                raise Exception('ELM initialize error')   
                
                    
    # This method just trains the ELM. If you wanna check the training error, set aval=True
    def train (self, aval=False, iteration=''):
        if iteration != '':
            iteration= str(iteration)
        p("Running training...")
        p("Computing the matrix H penroose")
        H = sigmoid(np.dot(self.inTrain, self.W))
        # Computing the weights beta
        p(self.inTrain)
        p(self.outTrain)
        self.beta = np.dot(np.linalg.pinv(H),self.outTrain)
        #print '\nCONDITION NUMBER:', np.linalg.cond(self.beta), '\n'

        if aval == True:
            outNet = np.dot (H,self.beta)
            if str(self.inputW) == 'uniform' or self.inputW is None or str(self.inputW)=='RO':
                print("inputW:", self.inputW)
                log("log/" + self.dataName + "/" + iteration + self.inputW + '.log', outNet)
            else:
                print("inputW:", 'RBM')
                log("log/" + self.dataName + "/" + iteration + "rbm.log" , outNet)

            miss = float(cont_error (self.outTrain, outNet))
            si = float(self.outTrain.shape[0])
            acc= (1-miss/si)*100
            print('Miss classification on the training: ', miss, ' of ', si, ' - Accuracy: ', acc, '%')
            return outNet, acc
            
    # This method executes the ELM, according to the weights and the data passed as parameter
    def getResult (self, data, realOutput=None, aval=False):
        # including 1 because the bias
        dataTest = np.concatenate ((data, np.ones([data.shape[0],1])), axis = 1)       
        # Getting the H matrix
        H = sigmoid (np.dot(dataTest, self.W))
        netOutput = np.dot (H, self.beta)
        if aval:
            miss = float(cont_error(realOutput, netOutput))
            si = float(netOutput.shape[0])
            acc = (1-miss/si)*100
            print('Miss classification on the test: ', miss, ' of ', si, ' - Accuracy: ',acc , '%')       
            return netOutput, acc
            
        return netOutput, None
        
    def getWeight(self):
        return self.W

    # This method saves the trained weights as a .csv file
    def saveELM (self, nameFile='ELM'):
        filename= nameFile + '-weightW.log'
        f= open(filename, "w")
        f.write(str(self.W))
        f.close
        print("(log)", filename)

        filename= nameFile + '-weightBeta.log'
        f= open(filename, "w")
        f.write(str(self.beta))
        f.close
        print("(log)", filename)

        # np.savetxt(nameFile+'-weightW.csv', self.W)
        # np.savetxt(nameFile+'-weightBeta.csv', self.beta)
        
    # This method computes the norm for input and output weights
    def getNorm (self, verbose=False):
        wNorm = np.linalg.norm(self.W)
        betaNorm = np.linalg.norm(self.beta)
        if verbose:
            print('The norm of W: ', wNorm)
            print('The norm of beta: ', betaNorm)
        return wNorm, betaNorm
        



     

      







