#/home/arissetyawan/anaconda3/bin/python3.6

import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
sys.path.insert (0, '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/')
from elm_train import *

DATA_DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/'

def read_csv(file, headers=None):
	df= pd.io.parsers.read_csv(file, header=headers)
	return df

def read_ssv(file, headers=None):
	df= pd.io.parsers.read_csv(file, sep=' ', header=headers)
	return df

# aris, sep 9
# df is pd.io.parsers.read_csv(csv_file)
#
def write_ssv(data, file_name, columns=None):
	klass=data.__class__.__name__
	if klass!='DataFrame':
		#print(columns)		
		data = pd.DataFrame(columns)
	else:
		data = data
	file_name= file_name.replace(".csv", ".ssv")
	data.to_csv(file_name, sep=' ', encoding='utf-8', header=None, index=False)
	print(file_name)
	return file_name

def write_csv(data, file_name, columns=None):
	klass=data.__class__.__name__
	if klass!='DataFrame':
		#print(columns)		
		data = pd.DataFrame(columns)
	else:
		data = data
	data.to_csv(file_name, sep=',', encoding='utf-8', index=False)
	print('Out: ', file_name)
	return file_name

def csv_to_ssv(file_name):
	df= pd.io.parsers.read_csv(file_name, header=None)
	file_name=  file_name.replace('.csv','.ssv')
	df.to_csv(file_name, sep=' ', encoding='utf-8', header=None, index=False)
	print('Out', file_name)
	return file_name

def ssv_to_csv(file_name):
	df= pd.io.parsers.read_csv(file_name, sep=' ', header=None)
	file_name=  file_name.replace('.csv','.ssv')
	df.to_csv(file_name, sep=',', encoding='utf-8', header=None, index=False)
	print('Out', file_name)
	return file_name

# arissetyawan, sep 9
# move column to front
# move_column_csv(DATA_DIABETES, ['A1','A2','A3','A4','A5','A6','A7','A8','Label'], 'Label');
def move_column_csv(file, attributes, col_name):
	print("\n\n -- Moving label to first column...")
	df = pd.io.parsers.read_csv(file, header=None, usecols=np.arange(len(attributes)))
	df.columns= attributes
	#print(df)
	cols = df.columns.tolist()
	n = int(cols.index(col_name))
	cols = [cols[n]] + cols[:n] + cols[n+1:]
	df = df[cols]
	df = df.drop(df.index[0])

	#print(df)
	new_file= file + '.label_first'
	write_csv(df, new_file)
	return new_file

def build_attributes(n):
	ret= []
	for i in range(n):
		ret.append('A' + str(i+1))
	ret.append('Label')
	return ret

def build_columns(data, n):
	ret= {}
	key= 'Label'
	for i in range(n):
		if i>0:
			key= 'A' + str(i+1)
		ret[key]= data[:, i]	
	return ret

# min max with 0..1
# input
# train = np.array([[ 1., -1.,  2.],
#                    [ 2.,  0.,  0.],
#                    [ 0.,  1., -1.]])
# normalize_min_max(train)
# output
# array([[ 0.5       ,  0.        ,  1.        ],
#        [ 1.        ,  0.5       ,  0.33333333],
#        [ 0.        ,  1.        ,  0.        ]])
#
def normalize_min_max(file_name):
	print("Normalizing...")
	df= read_csv(file_name)
	#print("Before norm min max", df)
	min_max_scaler = preprocessing.MinMaxScaler()
	
	df = df.drop(df.index[0])

	result= min_max_scaler.fit_transform(df)

	#print("After norm min max", result)
	return result

###############################################
###############################################



###############################################
DATA_DIABETIC= DATA_DIR + 'diabetic.csv'

attributes= build_attributes(19)
file_with_first_label= move_column_csv(DATA_DIABETIC, attributes, 'Label');

data = normalize_min_max(file_with_first_label)
columns= build_columns(data, 19)

file_name= file_with_first_label + '.min_max_scaler'
write_ssv(data, file_name, columns)

###############################################
DATA_IRIS= DATA_DIR + 'iris.csv'

attributes= build_attributes(4)
file_with_first_label= move_column_csv(DATA_IRIS, attributes, 'Label');

data = normalize_min_max(file_with_first_label)
columns= build_columns(data, 4)

file_name= file_with_first_label + '.min_max_scaler'
write_ssv(data, file_name, columns)


###############################################
DATA_AUSTRALIAN= DATA_DIR + 'australian.csv'

attributes= build_attributes(14)
file_with_first_label= move_column_csv(DATA_AUSTRALIAN, attributes, 'Label');

data = normalize_min_max(file_with_first_label)
columns= build_columns(data, 14)

file_name= file_with_first_label + '.min_max_scaler'
write_ssv(data, file_name, columns)

###############################################
DATA_AUSTRALIAN= DATA_DIR + 'isolet5.csv'
csv_to_ssv(DATA_AUSTRALIAN)