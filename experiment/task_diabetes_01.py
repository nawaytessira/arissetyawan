"""
Aris Setyawan
Sep 12. 2018
-----------------
This code is trying to investigate ELM standard
with # neurons
/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment/
/home/arissetyawan/anaconda3/bin/python3.6 task_diabetes_01.py
"""

from __future__ import print_function

import numpy as np
import time
import pandas as pd
import scipy.stats as stats
from elmr import ELMRandom 
from mltools import *


DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment/'
data= np.loadtxt(DIR + "pima-diabetes.txt.label_first.min_max")


acc_rand_train = []
acc_rand_test= []
tim_rand = []
acc_neurons= {}
tim_neurons= {}

neurons= [100, 200, 300, 400, 500, 600, 700, 800, 700, 800, 900, 1000]

folds= [0.7, 0.8 ] #train 
trials= 30

for t in range(trials):
	print(line(), "Trials: ", t+1)
	
	for i in neurons:
		print(line('-'), "Neurons: ", i)

		key= str(i)
		if not acc_neurons.get(key):
			acc_neurons[key]= []

		if not tim_neurons.get(key):
			tim_neurons[key]= []

		elmr = ELMRandom(["sigmoid", 1, i, True])
		init = time.time()

		for fold in folds:

			# split data in training and testing sets
			# use % training % testng and shuffle data before splitting
			print("\nK-Fold; Training:", fold, "Testing:", round(1-fold, 1))
			tr_set, te_set = split_sets(data, training_percent=fold, perm=True)

			tr_result = elmr.train(tr_set)
			tr_acc_rand= tr_result.get_accuracy()
			acc_rand_train.append(tr_acc_rand)

			print("Accuracy: ",tr_acc_rand)
			del(tr_acc_rand)
			del(tr_result)

		end = time.time()

		acc_neurons[key].append(np.asarray(acc_rand_train).mean())
		tim_neurons[key].append(end-init)

		acc_rand_train= []
		del(end)

elmr.print_parameters()

del(elmr)

all_neurons= {}
for k, v in acc_neurons.items():
	acc_neurons[k] = np.asarray(v)
	tim_neurons[k] = np.asarray(v)
	print('\nNeuron:', k)
	acc_mean = round(acc_neurons.get(k).mean(), 2)
	print('Accuracy -  mean: ', acc_mean, '| Std: ', acc_neurons.get(k).std())
	all_neurons[k]= acc_mean

all_times= {}
for k, v in tim_neurons.items():
	tim_neurons[k] = np.asarray(v)
	print('\nNeuron:', k)
	tim_neuron = round(tim_neurons.get(k).mean(), 2)
	print('Time - mean: ', tim_neurons.get(k).mean())
	all_times[k]= tim_neuron

print(all_neurons)
print(all_times)
print('Best Accuracy', max(all_neurons.items(), key = lambda x: x[1]))
print('Best Time', min(all_times.items(), key = lambda x: x[1]))

'''
print("\nT-Test\n======================")

print("Accuracy Random vs RBM")
df = pd.DataFrame({"random":acc_rand,
                   "rbm":acc_rbm,
                   "change":acc_rbm-acc_rand})
print(df.describe())
stat_result= stats.ttest_rel(a = acc_rand, b = acc_rbm)
significant_check(stat_result.pvalue)
'''