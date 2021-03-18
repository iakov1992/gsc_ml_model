import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.utils import shuffle
import os
import warnings
import gc
import time
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from optparse import OptionParser
warnings.filterwarnings("ignore")

def MaxScaler(df,train,column):
	vals = df[str(column)] / train[column].max()
	return vals.values

def MinMaxShiftedScaler(df,train,column):
	vals = (df[str(column)] - train[str(column)].min()) / (train[str(column)].max() - train[str(column)].min()) - 0.5
	return vals.values

def MinMaxScaler(df,train,column):
	vals = (df[str(column)] - train[str(column)].min()) / (train[str(column)].max() - train[str(column)].min())
	return vals.values

def StdScaler(df,train,column):
	vals = (df[str(column)] - train[str(column)].mean()) / train[str(column)].std()
	return vals.values


def ScaleFeatures(jet_type,etaLow,etaHigh,scale_target):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
		
	train_df = pd.read_csv(folder_name + "train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
	test_df = pd.read_csv(folder_name + "test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
	valid_df = pd.read_csv(folder_name + "valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
	
	if jet_type == "LCTopo":
		train_df = train_df[train_df['jet_ActiveArea4vec_eta'] < 100.]
		test_df = test_df[test_df['jet_ActiveArea4vec_eta'] < 100.]
		valid_df = valid_df[valid_df['jet_ActiveArea4vec_eta'] < 100.] 
		
	for col in train_df.columns:
		print("# of nans in train ",col,":",train_df[str(col)].isna().sum())
		print("# of nans in test ",col,":",test_df[str(col)].isna().sum())
		print("# of nans in valid ",col,":",valid_df[str(col)].isna().sum())
		
	train_df['jet_pt_response'] = train_df.jet_pt / train_df.jet_true_pt
	test_df['jet_pt_response'] = test_df.jet_pt / test_df.jet_true_pt
	valid_df['jet_pt_response'] = valid_df.jet_pt / valid_df.jet_true_pt
    
	features_max = []
	features_minmax = []
	for col in train_df.columns:
		for val in train_df[str(col)].values:
			if val < 0.:
				if col == "PtWeight":
					continue
				if col == "FlatWeight":
					continue
				features_minmax.append(str(col))
				break
	
	for col in train_df.columns:
		if str(col) in features_minmax:
			continue
		else:
			if col == "PtWeight":
				continue
			if col == "FlatWeight":
				continue
			features_max.append(str(col))
	'''		
	print("FEATURE MAX:")
	for i in features_max:
		print(i)
	print("FEATURE MINMAX:")
	for i in features_minmax:
		print(i)
	'''	
	scaled_train_df = pd.DataFrame()
	scaled_test_df = pd.DataFrame()
	scaled_valid_df = pd.DataFrame()
	
	scaled_train_df["PtWeight"] = train_df.PtWeight
	scaled_test_df["PtWeight"] = test_df.PtWeight
	scaled_valid_df["PtWeight"] = valid_df.PtWeight
	scaled_train_df["FlatWeight"] = train_df.FlatWeight
	scaled_test_df["FlatWeight"] = test_df.FlatWeight
	scaled_valid_df["FlatWeight"] = valid_df.FlatWeight
	
	for col in features_minmax:
		if scale_target == "no":
			if col == 'jet_pt_response':
				scaled_train_df[col] = train_df[col].values
				scaled_test_df[col] = test_df[col].values
				scaled_valid_df[col] = valid_df[col].values
			else:
				scaled_train_df[col] = MinMaxScaler(train_df,train_df,col)
				scaled_test_df[col] = MinMaxScaler(test_df,train_df,col)
				scaled_valid_df[col] = MinMaxScaler(valid_df,train_df,col)
		else:
			scaled_train_df[col] = MinMaxScaler(train_df,train_df,col)
			scaled_test_df[col] = MinMaxScaler(test_df,train_df,col)
			scaled_valid_df[col] = MinMaxScaler(valid_df,train_df,col)
			
	for col in features_max:
		if scale_target == "no":
			if col == 'jet_pt_response':
				scaled_train_df[col] = train_df[col].values
				scaled_test_df[col] = test_df[col].values
				scaled_valid_df[col] = valid_df[col].values
			else:
				scaled_train_df[col] = MaxScaler(train_df,train_df,col)
				scaled_test_df[col] = MaxScaler(test_df,train_df,col)
				scaled_valid_df[col] = MaxScaler(valid_df,train_df,col)
		else:
			scaled_train_df[col] = MaxScaler(train_df,train_df,col)
			scaled_test_df[col] = MaxScaler(test_df,train_df,col)
			scaled_valid_df[col] = MaxScaler(valid_df,train_df,col)
			
	for col in scaled_train_df.columns:
		print("# of nans in train ",col,":",scaled_train_df[str(col)].isna().sum())
		print("# of nans in test ",col,":",scaled_test_df[str(col)].isna().sum())
		print("# of nans in valid ",col,":",scaled_valid_df[str(col)].isna().sum())
		
	for col in scaled_train_df.columns:
		print "train:",col, "min:",scaled_train_df[col].min(),"max:",scaled_train_df[col].max()
		print "test:",col, "min:",scaled_test_df[col].min(),"max:",scaled_test_df[col].max()
		print "valid:",col, "min:",scaled_valid_df[col].min(),"max:",scaled_valid_df[col].max()
		
	print "Create csv files"
	if scale_target == "yes":
		scaled_train_df.to_csv(folder_name + "scaled_train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv",index=False)
		scaled_test_df.to_csv(folder_name + "scaled_test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv",index=False)
		scaled_valid_df.to_csv(folder_name + "scaled_valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv",index=False)
	else:
		scaled_train_df.to_csv(folder_name + "scaled_train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_targetIsNotScaled_noNan.csv",index=False)
		scaled_test_df.to_csv(folder_name + "scaled_test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_targetIsNotScaled_noNan.csv",index=False)
		scaled_valid_df.to_csv(folder_name + "scaled_valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_targetIsNotScaled_noNan.csv",index=False)
	


parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--scale_target', default = "yes", type = "string", help = "scale target variable (yes or no)")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
ScaleFeatures(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh,scale_target=opt.scale_target)