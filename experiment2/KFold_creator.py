# -*- coding: utf-8 -*-
"""

Author: arissetyawan.email@gmail.com
"""

MAIN_DIR= "/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/"
DATA_PATH= MAIN_DIR + "datasets/"

import sys
import pandas as pd
import numpy as np
from dataManipulation import *
sys.path.insert (0, MAIN_DIR)
sys.path.insert (0, "/usr/local/lib/python3.6/site-packages/")

if len(sys.argv)!= 4:
    print("Please pass argument for dataset name and fold number and label position eg. iris 5 last")
else:
    dataset= sys.argv[1]
    K= int(sys.argv[2])
    postOut= sys.argv[3]
    print("Dataset with K split", dataset, K)

    objFold= Fold(dataset, K, postOut)
    objFold.stratified()