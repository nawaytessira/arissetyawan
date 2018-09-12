# -*- coding: utf-8 -*-
"""
Author: Aris Setyawan
E-mail: arissetyawan.email@gmail.com

If you find some bug, plese e-mail me =)

"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert (0, '/usr/local/lib/python3.6/site-packages/')
import tensorflow as tf
from rbm import *
from rbm_tensorflow import *
'''
As we will see, each row has 785 columns, with the first being the label and the rest of them representing the pixel values (28x28) of the image.
'''
train = pd.read_csv("datasets/mnist/mnist_train.csv")
print(train.head())

'''
Next, we will need to separate the labels from the pixel values.
'''
x_train = train.iloc[:, 1:].values.astype('float32')
labels = train.iloc[:, 0].values.astype('int32')

'''
Let's plot the first 5 images from the dataset to better visualize the data.
fig = plt.figure(figsize=(12, 12))
for i in range(5)
    fig.add_subplot(1, 5, i+1)
    plt.title('Label: {label}'.format(label=labels[i]))
    plt.imshow(x_train[i].reshape(28, 28), cmap='Greys')
    plt.show()
'''

'''
Since this is a multiclass classification problem, we will One Hot Encode the labels. 
This simply means that we will use vectors to represent each class, instead of the label value. 
Each vector contains the value 1 at the index corresponding to the class it represents, 
with the rest of the values set to 0.
'''
CLASSES = 10
y_train = np.zeros([labels.shape[0], CLASSES])
for i in range(labels.shape[0]):
    y_train[i][labels[i]] = 1
y_train.view(type=np.matrix)

'''
The next step is to split the data into training and testing parts, 
since we would like to test our accuracy of our model at the end. 
We will use around 10% of our training data for testing.
Now, our data is ready for both training and testing our neural network. 
Next, we will take a look at the implementation of the Extreme Learning Machine.
'''
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
print('Train size: {train}, Test size: {test}'.format(train=x_train.shape[0], test=x_test.shape[0]))

'''
Let's start by defining some constants and generate the input to hidden layer weights:
'''
INPUT_LENGHT = x_train.shape[1] # 784 
print(INPUT_LENGHT)
hidden_units = [6000, 7000, 8000, 9000, 10000]


'''
RMBnet
maxIterRbm= 17
rbmNet = RBM_TF (dataIn=x_train, numHid=HIDDEN_UNITS, rbmType='GBRBM')
rbmNet.train (maxIter=maxIterRbm, lr=0.01, wc=0.01, iMom=0.5, fMom=0.9, cdIter=1, batchSize=250, freqPrint=10)
W = np.concatenate ((rbmNet.getWeights(), rbmNet.getHidBias()), axis = 0)        
del(rbmNet)    

'''


'''
The next step is to compute our hidden layer to output weights. This is done in the following way:
    Compute the dot product between the input and input-to-hidden layer weights, 
    and apply some activation function. Here we will use ReLU, 
    since it is simple and in this case it gives us a good result:
'''
def input_to_hidden(x):
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a) # ReLU
    return a


for hidden_unit in hidden_units:
	Win = np.random.normal(size=[INPUT_LENGHT, hidden_unit])
	print('Input Weight shape: {shape}'.format(shape=Win.shape))
	print(Win)

	'''
	Compute output weights, this is a standard least square error regression problem,
	since we try to minimize the least square error between the predicted labels and
	the training labels. The solution to this is:
	'''
	print("input_to_hidden...")
	X = input_to_hidden(x_train)

	print("transposing...")
	Xt = np.transpose(X)

	print("dot operation...")
	Wout = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
	print('Output weights shape: {shape}'.format(shape=Wout.shape))

	'''
	Now that we have our trained model, let's create a function that predicts the output, 
	this is done simply by computing the dot product between the result from 
	the input_to_hidden function we defined earlier, with the output weights:
	'''
	def predict(x):
	    x = input_to_hidden(x)
	    y = np.dot(x, Wout)
	    return y

	'''
	Next, we can test our model:
	'''
	y = predict(x_test)
	correct = 0
	total = y.shape[0]

	digits = len(str(total - 1))
	delete = "\b" * (digits + 1)

	for i in range(total):
		print("{0}{1:{2}}".format(delete, i, digits), end="")
		sys.stdout.flush()

		predicted = np.argmax(y[i])
		test = np.argmax(y_test[i])
		correct = correct + (1 if predicted == test else 0)

	print('SUMMARIES')
	print('Train size: {train}, Test size: {test}'.format(train=x_train.shape[0], test=x_test.shape[0]))
	print('Input Weight shape: {shape}'.format(shape=Win.shape))
	print(Win)
	print('Output weights shape: {shape}'.format(shape=Wout.shape))
	print('Accuracy: {:f}'.format(correct/total))
