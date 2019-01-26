#/home/arissetyawan/anaconda3/bin/python3.6

import pandas as pd
import numpy as np

DATA_DIR= '/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment/'
DIABETES= DATA_DIR + 'pima-diabetes.csv'
df = pd.io.parsers.read_csv(DIABETES,
     header=None, usecols=[0,1,2,3,4,5,6,7,8])

attributes= ['Attr1','Attr2','Attr3','Attr4','Attr5','Attr6','Attr7','Attr8']
df.columns= attributes + ['Label']
print(df.columns)
print(df)

from sklearn import preprocessing


minmax_scale = preprocessing.MinMaxScaler().fit(df[attributes])
df_minmax = minmax_scale.transform(df[attributes])

print(df_minmax)