import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
import os
import warnings
import gc
import time
from sklearn import metrics
warnings.filterwarnings("ignore")

print "Read dataframe"
train_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_train_df_standardScaler.csv")
test_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_test_df_standardScaler.csv")
valid_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_valid_df_standardScaler.csv")
print "train_df is read"

FlatWeight_train = train_df['FlatWeight']
FlatWeight_test = test_df['FlatWeight']
FlatWeight_valid = valid_df['FlatWeight']

jet_true_pt_train = train_df['jet_true_pt']
jet_true_pt_test = test_df['jet_true_pt']
jet_true_pt_valid = valid_df['jet_true_pt']

PtWeight_train = train_df['PtWeight']
PtWeight_test = test_df['PtWeight']
PtWeight_valid = valid_df['PtWeight']

X = train_df.drop(['jet_true_pt','FlatWeight','PtWeight'],axis=1)
y = train_df.jet_true_pt

print "Select Features"
'''
model = XGBRegressor(learning_rate=1.e-4,n_jobs=-1,random_state=17)
sfm = SelectFromModel(model,threshold="median")
sfm.fit(X, y)
train = sfm.transform(X)
feature_idx = sfm.get_support()
'''

skb = SelectKBest(f_regression, k=50)
skb.fit(X,y)
feature_idx = skb.get_support()

print "feature_idx = ", feature_idx
feature_name = X.columns[feature_idx]
print "feature_name: ",feature_name
print "Selected Features:"
for col in feature_name:
	print col

for col in train_df.columns:
	if col in feature_name:
		continue
	else:
		train_df.drop(str(col),axis=1,inplace=True)
		test_df.drop(str(col),axis=1,inplace=True)
		valid_df.drop(str(col),axis=1,inplace=True)

print "Best Features:"
for col in train_df.columns:
	print col

train_df['jet_true_pt'] = jet_true_pt_train
test_df['jet_true_pt'] = jet_true_pt_test
valid_df['jet_true_pt'] = jet_true_pt_valid

train_df['PtWeight'] = PtWeight_train
test_df['PtWeight'] = PtWeight_test
valid_df['PtWeight'] = PtWeight_valid

train_df['FlatWeight'] = FlatWeight_train
test_df['FlatWeight'] = FlatWeight_test
valid_df['FlatWeight'] = FlatWeight_valid

train_df.to_csv('/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_train_df_standardScaler_50BestFeatures.csv',index=False)
test_df.to_csv('/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_test_df_standardScaler_50BestFeatures.csv',index=False)
valid_df.to_csv('/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_valid_df_standardScaler_50BestFeatures.csv',index=False)


