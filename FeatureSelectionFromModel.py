import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
import os
import warnings
import gc
import time
from sklearn import metrics
warnings.filterwarnings("ignore")
from optparse import OptionParser

def SelectionFromModel(jet_type,etaLow,etaHigh):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
	
	trainName = folder_name + 'train_etaLow'+str(etaLow)+'_etaHigh'+str(etaHigh)+'.csv'
	train_df = pd.read_csv(trainName)
	FlatWeight = train_df['FlatWeight']
	PtWeight = train_df['PtWeight']
	train_df['jet_pt_response'] = train_df.jet_pt / train_df.jet_true_pt
	y = train_df.jet_pt_response
	X = train_df.drop(['PtWeight', 'FlatWeight','jet_true_eta','jet_true_phi','jet_true_e',
					   'jet_respE', 'jet_respPt','jet_true_m', 'jet_JESE','jet_JESPt','jet_JESEta',
					   'jet_JESPhi','jet_JESMass','jet_true_pt','jet_pt_response'],axis=1)


	'''
	model = RandomForestRegressor(n_estimators = 300,criterion='mse',max_depth=3,
								n_jobs=20,random_state=17,verbose=4)
	'''
	model = GradientBoostingRegressor(n_estimators = 100,loss='ls',max_depth=3,learning_rate=0.01,
									random_state=17,verbose=4)
	'''
	model = LinearRegression(n_jobs=20)
	'''
	model = model.fit(X,y,sample_weight=FlatWeight.values)

	print model.feature_importances_

	sfm = SelectFromModel(model,prefit=True,threshold="median")
	feature_idx = sfm.get_support()
	print "feature_idx = ", feature_idx
	feature_name = X.columns[feature_idx]

	print "List Of Best Features for",jet_type,"etaLow = ",etaLow,"etaHigh =", etaHigh, ":"
	for col in feature_name:
		print col



parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
SelectionFromModel(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh)