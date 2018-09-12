#/home/arissetyawan/anaconda3/bin/python3.6

import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
sys.path.insert (0, '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment/')
from elm_train import *

DATA_DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment/'

def read_csv(file, headers):
	df= pd.io.parsers.read_csv(file, header=headers)
	return df

def read_plain_csv(file):
	df= pd.io.parsers.read_csv(file, header=None)
	return df

# aris, sep 9
# df is pd.io.parsers.read_csv(csv_file)
#
def write_ssv(data, file_name, columns=None):
	klass=data.__class__.__name__
	if klass!='DataFrame':
		print(columns)		
		data = pd.DataFrame(columns)
	else:
		data = data
	data.to_csv(file_name, sep=' ', encoding='utf-8', index=False)
	print(file_name)
	return file_name


# arissetyawan, sep 9
# move column to front
# move_column_csv(DATA_DIABETES, ['Attr1','Attr2','Attr3','Attr4','Attr5','Attr6','Attr7','Attr8','Label'], 'Label');
def move_column_csv(file, attributes, col_name):
	print("Moving label to first column...")
	df = pd.io.parsers.read_csv(file, header=None, usecols=np.arange(len(attributes)))
	df.columns= attributes
	print("Before moving...")
	print(df)
	cols = df.columns.tolist()
	n = int(cols.index(col_name))
	cols = [cols[n]] + cols[:n] + cols[n+1:]
	df = df[cols]
	df = df.drop(df.index[0])

	print("After moving...")
	print(df)
	new_file= file + '.label_first'
	write_csv(df, new_file)
	print('Wrote new file: ', new_file)
	return new_file

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
	min= 0
	max= 1
	df= read_plain_csv(file_name)
	print("Before norm min max", df)
	min_max_scaler = preprocessing.MinMaxScaler()
	
	df = df.drop(df.index[0])

	result= min_max_scaler.fit_transform(df)

	print("After norm min max", result)
	return result

###############################################
###############################################
DATA_DIABETES= DATA_DIR + 'pima-diabetes.csv'

attributes= ['Attr1','Attr2','Attr3','Attr4','Attr5','Attr6','Attr7','Attr8','Label']
file_with_first_label= move_column_csv(DATA_DIABETES, attributes, 'Label');

data = normalize_min_max(file_with_first_label)
columns= {	
			'Label': data[:,0],
			'Attr1': data[:,1],
			'Attr2': data[:,2],
			'Attr3': data[:,3],
			'Attr4': data[:,4],
			'Attr5': data[:,5],
			'Attr6': data[:,6],
			'Attr7': data[:,7],
			'Attr8': data[:,8]
		}

file_name= file_with_first_label +  '.min_max_scaler'
write_csv(data, file_name, columns)

###############################################
DATA_IRIS= DATA_DIR + 'iris.csv'

attributes= ['Attr1','Attr2','Attr3','Attr4','Label']
file_with_first_label= move_column_csv(DATA_IRIS, attributes, 'Label');

data = normalize_min_max(file_with_first_label)
columns= {	
			'Label': data[:,0],
			'Attr1': data[:,1],
			'Attr2': data[:,2],
			'Attr3': data[:,3],
			'Attr4': data[:,4],
		}

file_name= file_with_first_label +  '.min_max_scaler'
write_csv(data, file_name, columns)



