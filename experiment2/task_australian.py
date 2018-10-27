from __future__ import print_function

import numpy as np
import time
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA, KernelPCA
from elmr import ELMRandom 
from rbm import RBM 
from mltools import *

def significant_check(pvalue):
	alpha = 0.05
	if pvalue > alpha:
		print('Not significant >=', alpha, pvalue)
	else:
		print('Significant < ', alpha, pvalue)


def rbm_weigthing(training, neurons, maxIter=100, lr=0.0001, wc=0.01, iMom=0.5, fMom=0.9, cdIter=1, batchSize=250, freqPrint=10):
	rbmNet = RBM(dataIn=training, numHid=neurons)
	rbmNet.train (maxIter=maxIter, lr=lr, wc=wc, iMom=iMom, fMom=fMom, cdIter=cdIter, batchSize=batchSize, freqPrint=freqPrint)
	W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)
	return W

def random_orthogonal(training, neurons):
    if neurons >= training.shape[1]:
        W = np.random.uniform(-1,1,[training.shape[1]+1,neurons])
        W,_ = np.linalg.qr(W.T)
        W = W.T
    else:   
        print('Starting PCA...')
        A = np.random.uniform(-1,1,[neurons,neurons])
        A,_ = np.linalg.qr(A.T)
        A = A.T
        pca = PCA(n_components=neurons)                   
        wpca = pca.fit_transform(training.T).T
        """
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        wpca = kpca.fit_transform(training.T)
		"""
        W = np.dot(A,wpca).T                        
        # including the bias
        b = np.random.uniform(-1,1,[1,W.shape[1]])                        
        W = np.vstack((W,b))   
    return W


DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment/'
data = np.loadtxt(DIR + "australian.ssv.label_first.min_max_scaler")

neurons= 100
# search for best parameter for this dataset
# define "kfold" cross-validation method, "accuracy" as a objective function
# to be optimized and perform 10 searching steps.
# best parameters will be saved inside 'elmk' object
# elmr.search_param(data, cv="kfold", of="accuracy", eval=10)


# elmr.print_parameters()
# split data in training and testing sets
# use 80% of dataset to training and shuffle data before splitting
tr_set, te_set = split_sets(data, training_percent=.7, perm=True)

# train and test
# results are Error objects

acc_rand = list()
tim_rand = list()

acc_ortho = list()
tim_ortho = list()

acc_rbm = list()
tim_rbm = list()

trial= 10


for i in range(trial):

	print("\nTrial: ", i + 1)

	elmr = ELMRandom(["sigmoid", 1, neurons, False, []])

	elmo = ELMRandom(["sigmoid", 1, neurons, False, random_orthogonal(data, neurons)])

	rbm = rbm_weigthing(data, 35, maxIter=1, lr=0.01, wc=0.1, iMom=0.1, fMom=0.1, cdIter=1, batchSize=150, freqPrint=10)
	elmrbm = ELMRandom(["sigmoid", 1, neurons, False, rbm])

	# ELM random
	init = time.time()
	tr_result1 = elmr.train(tr_set)
	tr_acc_rand= tr_result1.get_accuracy()
	#te_result = elmr.test(te_set)
	#print(te_result.get_accuracy())
	end = time.time()
	time1= end-init
	acc_rand.append(tr_acc_rand)
	tim_rand.append(time1)

	# ELM orthogonal
	init = time.time()
	tr_result2 = elmo.train(tr_set)
	tr_acc_ortho= tr_result2.get_accuracy()
	#te_result = elmr.test(te_set)
	#print(te_result.get_accuracy())
	end = time.time()
	time2= end-init
	acc_ortho.append(tr_acc_ortho)
	tim_ortho.append(time2)

	# ELM rbm
	init = time.time()
	tr_result3 = elmrbm.train(tr_set)
	tr_acc_rbm = tr_result3.get_accuracy()
	#te_result = elmr.test(te_set)
	#print(te_result.get_accuracy())
	end = time.time()
	time3= end-init
	acc_rbm.append(tr_acc_rbm)
	tim_rbm.append(time3)

	del(tr_result1)
	del(tr_result2)
	del(tr_result3)


	print("Accuracy, Time: ")
	print("ELM Random: ", tr_acc_rand, time1)
	print("ELM Orthogonal: ", tr_acc_ortho, time2)
	print("ELM RBM: ", tr_acc_rbm, time3)

	#perc = float(i) / trial * 100
	#print("Training % complete {}".format(perc), end='\r\r', flush=True)



del(elmr)
del(elmo)
del(elmrbm)

print ('ELM Random:')
acc_rand = np.asarray(acc_rand)
tim_rand = np.asarray(tim_rand)
print ('Accuracy -  mean: ', acc_rand.mean(), '| Std: ', acc_rand.std())
print ('Time - mean: ', tim_rand.mean(), ' | Std: ', tim_rand.std())


print ('ELM Orthogonal:')
acc_ortho = np.asarray(acc_ortho)
tim_rand = np.asarray(tim_rand)
print ('Accuracy -  mean: ', acc_ortho.mean(), '| Std: ', acc_ortho.std())
print ('Time - mean: ', tim_ortho.mean(), ' | Std: ', tim_ortho.std())

print ('ELM RBM:')
acc_rbm = np.asarray(acc_rbm)
tim_rbm = np.asarray(tim_rbm)
print ('Accuracy -  mean: ', acc_rbm.mean(), '| Std: ', acc_rbm.std())
print ('Time - mean: ', tim_rbm.mean(), ' | Std: ', tim_rbm.std())


print("\nT-Test\n======================")
print("Random vs RBM")
df = pd.DataFrame({"random":acc_rand,
                   "rbm":acc_rbm,
                   "change":acc_rbm-acc_rand})

print(df.describe())
stat_result= stats.ttest_rel(a = acc_rand, b = acc_rbm)
significant_check(stat_result.pvalue)

print("Orthogonal vs RBM")
df = pd.DataFrame({"orthogonal":acc_ortho,
                   "rbm":acc_rbm,
                   "change":acc_rbm-acc_ortho})

print(df.describe())
stat_result= stats.ttest_rel(a = acc_ortho, b = acc_rbm)
significant_check(stat_result.pvalue)