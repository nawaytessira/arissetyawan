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
		print('[X] Not significant >=', alpha, pvalue)
	else:
		print('[V] Significant < ', alpha, pvalue)


def rbm(training_array, neurons):
	maxIterRbm= 100
	RBMneurons= 300
	rbmNet = RBM(dataIn=training_array, numHid=RBMneurons)
	rbmNet.train (maxIter=maxIterRbm, lr=0.01, wc=0.01, iMom=0.5, fMom=0.9, cdIter=1, batchSize=250, freqPrint=10)
	W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)
	return W

def random_orthogonal(training_array, neurons):
    if neurons >= training_array.shape[1]:
        W = np.random.uniform(-1,1,[training_array.shape[1]+1,neurons])
        W,_ = np.linalg.qr(W.T)
        W = W.T
    else:   
        print('Starting PCA...')
        A = np.random.uniform(-1,1,[neurons,neurons])
        A,_ = np.linalg.qr(A.T)
        A = A.T
        pca = PCA(n_components=neurons)                   
        wpca = pca.fit_transform(training_array.T).T
        """
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        wpca = kpca.fit_transform(training_array.T)
		"""
        W = np.dot(A,wpca).T                        
        # including the bias
        b = np.random.uniform(-1,1,[1,W.shape[1]])                        
        W = np.vstack((W,b))   
    return W


DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/'
# load dataset
data= np.loadtxt(DIR + "iris.csv")

neurons= 1000

acc_rand = list()
tim_rand = list()
rmse_rand = list()

acc_ortho = list()
tim_ortho = list()
rmse_ortho = list()

acc_rbm = list()
tim_rbm = list()
rmse_rbm = list()

trial= 10

total_fold= 10


for i in range(trial):

	for fold in range(total_fold):

		print("\n************* Trial: ", i + 1)
		# split data in training and testing sets
		# use 80% of dataset to training and shuffle data before splitting
		fold_value = round((fold+1)/float(total_fold) , 1)
		print("K-Fold; Training:", fold_value, "Testing:", round(1-fold_value, 1))
		tr_set, te_set = split_sets(data, training_percent=fold_value, perm=True)

		w_rbm = rbm(data, 100)
		elmrbm = ELMRandom(["sigmoid", 1, neurons, False, w_rbm])
		elmr = ELMRandom(["sigmoid", 1, neurons, False])
		elmo = ELMRandom(["sigmoid", 1, neurons, False, random_orthogonal(data, neurons)])

		# ELM random
		init = time.time()
		tr_result = elmr.train(tr_set)
		tr_acc_rand= tr_result.get_accuracy()
		#te_result = elmr.test(te_set)
		#print(te_result.get_accuracy())
		end = time.time()
		acc_rand.append(tr_acc_rand)
		tim_rand.append(end-init)

		# ELM orthogonal
		init = time.time()
		tr_result = elmo.train(tr_set)
		tr_acc_ortho= tr_result.get_accuracy()
		#te_result = elmr.test(te_set)
		#print(te_result.get_accuracy())
		end = time.time()
		acc_ortho.append(tr_acc_ortho)
		tim_ortho.append(end-init)

		# ELM rbm
		init = time.time()
		tr_result = elmrbm.train(tr_set)
		tr_acc_rbm = tr_result.get_accuracy()
		#te_result = elmr.test(te_set)
		#print(te_result.get_accuracy())
		end = time.time()
		acc_rbm.append(tr_acc_rbm)
		tim_rbm.append(end-init)

		del(tr_result)


		print("Accuracy: ")
		print("ELM Random: ", tr_acc_rand)
		print("ELM Orthogonal: ", tr_acc_ortho)
		print("ELM RBM: ", tr_acc_rbm)


	#perc = float(i) / trial * 100
	#print("Training % complete {}".format(perc), end='\r\r', flush=True)


elmr.print_parameters()
elmo.print_parameters()
elmrbm.print_parameters()

del(elmr)
del(elmo)
del(elmrbm)

print ('ELM Random:')
acc_rand = np.asarray(acc_rand)
tim_rand = np.asarray(tim_rand)
rmse_rand = np.asarray(rmse_rand)
print ('Accuracy -  mean: ', acc_rand.mean(), '| Std: ', acc_rand.std())
print ('Time - mean: ', tim_rand.mean(), ' | Std: ', tim_rand.std())


print ('ELM Orthogonal:')
acc_ortho = np.asarray(acc_ortho)
tim_rand = np.asarray(tim_rand)
rmse_ortho = np.asarray(rmse_ortho)
print ('Accuracy -  mean: ', acc_ortho.mean(), '| Std: ', acc_ortho.std())
print ('Time - mean: ', tim_rand.mean(), ' | Std: ', tim_rand.std())

print ('ELM RBM:')
acc_rbm = np.asarray(acc_rbm)
tim_rbm = np.asarray(tim_rbm)
rmse_rbm = np.asarray(rmse_rbm)
print ('Accuracy -  mean: ', acc_rbm.mean(), '| Std: ', acc_rbm.std())
print ('Time - mean: ', tim_rbm.mean(), ' | Std: ', tim_rbm.std())


print("\nT-Test\n======================")

print("Accuracy Random vs RBM")
df = pd.DataFrame({"random":acc_rand,
                   "rbm":acc_rbm,
                   "change":acc_rbm-acc_rand})
print(df.describe())
stat_result= stats.ttest_rel(a = acc_rand, b = acc_rbm)
significant_check(stat_result.pvalue)

print("Accuracy Orthogonal vs RBM")
df = pd.DataFrame({"orthogonal":acc_ortho,
                   "rbm":acc_rbm,
                   "change":acc_rbm-acc_ortho})
print(df.describe())
stat_result= stats.ttest_rel(a = acc_ortho, b = acc_rbm)
significant_check(stat_result.pvalue)

"""
print("RMSE Random vs RBM")
df = pd.DataFrame({"random":rmse_rand,
                   "rbm":rmse_rbm,
                   "change":rmse_rbm-rmse_rand})
print(df.describe())
stat_result= stats.ttest_rel(a = rmse_rand, b = rmse_rbm)
significant_check(stat_result.pvalue)

print("RMSE Orthogonal vs RBM")
df = pd.DataFrame({"orthogonal":rmse_ortho,
                   "rbm":rmse_rbm,
                   "change":rmse_rbm-rmse_ortho})
print(df.describe())
stat_result= stats.ttest_rel(a = rmse_ortho, b = rmse_rbm)
significant_check(stat_result.pvalue)
"""