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
from optparse import OptionParser
warnings.filterwarnings("ignore")


def SelectBestFeatures(jet_type,etaLow,etaHigh):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
	
	trainName = folder_name + 'train_etaLow'+str(etaLow)+'_etaHigh'+str(etaHigh)+'_noNan.csv'
	print "Read File:",trainName
	train_df = pd.read_csv(trainName,nrows=500000)
	FlatWeight = train_df['FlatWeight']
	PtWeight = train_df['PtWeight']
	train_df['jet_pt_response'] = train_df.jet_pt / train_df.jet_true_pt
	y = train_df.jet_pt_response
	X = train_df.drop(['PtWeight', 'FlatWeight','jet_true_eta','jet_true_phi','jet_true_e',
					   'jet_respE', 'jet_respPt','jet_true_m', 'jet_JESE','jet_JESPt','jet_JESEta',
					   'jet_JESPhi','jet_JESMass','jet_true_pt','jet_pt_response'],axis=1)

	select_nfeatures = 30
	skb = SelectKBest(f_regression, k=select_nfeatures)
	#skb = SelectKBest(mutual_info_regression, k=select_nfeatures)
	skb.fit(X,y)
	feature_idx = skb.get_support()

	print "feature_idx = ", feature_idx
	feature_name = X.columns[feature_idx]
	print "feature_name: ",feature_name
	print "Selected Features for",jet_type,"jets with etaLow = ",etaLow,"etaHigh =",etaHigh,":"
	for col in feature_name:
		print col

parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
SelectBestFeatures(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh)