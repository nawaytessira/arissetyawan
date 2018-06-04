

import sys

# We’ll use NumPy matrices in TensorFlow
import numpy as np 

sys.path.insert (0, '/usr/local/lib/python3.6/site-packages/')
import tensorflow as tf


W = np.concatenate(([2,2],[1,1,2]))        
print(W)


# Define a 2x2 matrix in 3 different ways
m1 = [[1.0, 2.0], 
      [3.0, 4.0]]
m2 = np.array([[1.0, 2.0], 
               [3.0, 4.0]], dtype=np.float32)
m3 = tf.constant([[1.0, 2.0], 
                  [3.0, 4.0]])

# Print the type for each matrix
print(type(m1))
print(type(m2))
print(type(m3))

# Create tensor objects out of the different types
t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

# Notice that the types will be the same now
print(type(t1))
print(type(t2))
print(type(t3))



import sys
sys.path.insert (0, '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/__THESIS___/ELM/codes/rbm-elm/')
sys.path.insert (0, '/usr/local/lib/python3.6/site-packages/')

import tensorflow as tf
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
import scipy
import plotly.plotly as py

from elm import *
from elm_tensorflow import *
from utilsClassification import *
from dataManipulation import data
from rbm import *
from rbm_tensorflow import *

from plotly.tools import FigureFactory as FF

it = 100
hidNeurons = 250
maxIterRbm = 17
 
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
DATA_PATH= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/__THESIS___/ELM/codes/rbm-elm/datasets'
#     def __init__(self, dataset=None, train=None, val=None, test=None, percTrain=0.7, percVal=0.1, percTest=0.2, normType='max', shuf=True, posOut='last', outBin=True):
diabetes_train = np.genfromtxt(DATA_PATH + '/diabetic/diabetes_train', delimiter=' ')
print(diabetes_train)

dataset= data(dataset=diabetes_train, percTrain=0.7, percVal=0.1, percTest=0.2, normType='max', shuf=True, posOut='last', outBin=True)
print(dataset)
dataset.load(trainIn=dataset.trainIn) #, trainOut=None, valIn=None, valOut=None, testIn=None, testOut=None):
dataset.save(name=DATA_PATH + '/diab', ext='.ssv')
