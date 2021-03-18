import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import gc
import time
from sklearn import metrics
from scipy.stats import ks_2samp

np.set_printoptions(precision=7)
warnings.filterwarnings("ignore")

print "Read dataframe"
train_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_train_df_eta02_20200520.csv")
test_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_test_df_eta02_20200520.csv")
FlatWeight = train_df['FlatWeight']
PtWeight = train_df['PtWeight']

X_train = train_df.drop(['jet_true_pt','FlatWeight','PtWeight'],axis=1)
y_train = train_df.jet_true_pt

X_test = train_df.drop(['jet_true_pt','FlatWeight','PtWeight'],axis=1)
y_test = train_df.jet_true_pt

print "Check the similarity for train and test"
for col in X_train.columns:
	print "KS Statistics for the feature",col,":",ks_2samp(train_df[str(col)],test_df[str(col)])	


print "Check the features similarity for train"
for col in X_train.columns:
	for col2 in X_train.columns:
		if col == col2:
			continue
		print "KS Statistics for the feature",col," with",col2,":",ks_2samp(train_df[str(col)],train_df[str(col2)])
