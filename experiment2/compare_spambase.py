# log/spambaseelmrnd-acc.20190121-113539
# log/spambaseelmro-acc.20190121-113539
# log/spambaseelmrbm-acc.20190121-113539
# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

Revised: Aris Setyawan
E-mail: arissetyawan.email@gmail.com

"""

MAIN_DIR= "/media/arissetyawan/01D01F7DA71A34F01/__PASCA__/_THESIS_/experiment2/"
DATA_PATH= MAIN_DIR + "datasets/"

import sys
sys.path.insert (0, MAIN_DIR)
sys.path.insert (0, "/usr/local/lib/python3.6/site-packages/")

dname1= 'log/spambaseelmrnd-acc.20190121-113539'
dname2= 'log/spambaseelmro-acc.20190121-113539'
dname3= 'log/spambaseelmrbm-acc.20190121-113539'

# ----------------------------------------------------
text_file = open(dname1, "r")
data1 = text_file.read()

text_file = open(dname2, "r")
data2 = text_file.read()

text_file = open(dname3, "r")
data3 = text_file.read()

# ----------------------------------------------------
import re
from ast import literal_eval

data1= re.sub("\s+", ",", data1.strip())
data1= literal_eval(data1)

data2= re.sub("\s+", ",", data2.strip())
data2= literal_eval(data2)

data3= re.sub("\s+", ",", data3.strip())
data3= literal_eval(data3)

# ----------------------------------------------------
from scipy.stats import friedmanchisquare
print('------------------------')
print("FRIEDMAN")
# compare samples
stat, p = friedmanchisquare(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')

# ----------------------------------------------------

from scipy.stats import wilcoxon
print('------------------------')
print("WILCOXON")
# compare samples
stat, p = wilcoxon(data1, data3)
print('Statistics=%.6f, p=%.6f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('RBM vs RND: Same distribution (fail to reject H0)')
else:
	print('RBM vs RND: Different distribution (reject H0)')

stat, p = wilcoxon(data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('RBM vs RO: Same distribution (fail to reject H0, sample equal)')
else:
	print('RBM vs RO: Different distribution (reject H0)')

# ----------------------------------------------------

from scipy.stats import mannwhitneyu
print('------------------------')
print("MANNWHITNEYU")
# compare samples
stat, p = mannwhitneyu(data1, data3)
print('Statistics=%.6f, p=%.6f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('RBM vs RND: Same distribution (fail to reject H0)')
else:
	print('RBM vs RND: Different distribution (reject H0)')

stat, p = mannwhitneyu(data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('RBM vs RO: Same distribution (fail to reject H0, sample equal)')
else:
	print('RBM vs RO: Different distribution (reject H0)')
# ----------------------------------------------------

from scipy import stats
print('------------------------')
print("T-Test")
t, p = stats.ttest_ind(data1, data3)
# print("t = ", round(t2, 6))
# interpret
alpha = 0.05
if p > alpha:
	print('RBM vs RND: Same distribution (fail to reject H0, sample equal)')
else:
	print('RBM vs RND: Different distribution (reject H0)')

t, p = stats.ttest_ind(data2, data3)
# interpret
alpha = 0.05
if p > alpha:
	print('RBM vs RO: Same distribution (fail to reject H0, sample equal)')
else:
	print('RBM vs RO: Different distribution (reject H0)')
