import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras 
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import h5py
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import warnings
import gc
import time
from sklearn import metrics
from optparse import OptionParser
warnings.filterwarnings("ignore")


def mape(y_true,y_pred,sample_weight):
	err = 0.
	total_weight = 0.
	for i in range(len(y_true)):
		sample_error = np.abs(sample_weight[i]*(y_true[i] - y_pred[i])/y_true[i])
		err += sample_error
		total_weight += sample_weight[i]
	wmape = 100 * err / total_weight
	return wmape

def MakeNNPredictions(jet_type,etaLow,etaHigh,scale_target,variable_type):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
		if scale_target == "no":
			folder_name = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/'
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
		if scale_target == "no":
			folder_name = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/'

	fileNameTrain = folder_name + "scaled_train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv"
	fileNameTest = folder_name + "scaled_test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv"
	fileNameValid = folder_name + "scaled_valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv"
	if scale_target == "no":
		fileNameTrain = folder_name + "scaled_train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_targetIsNotScaled_noNan.csv"
		fileNameTest = folder_name + "scaled_test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_targetIsNotScaled_noNan.csv"
		fileNameValid = folder_name + "scaled_valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_targetIsNotScaled_noNan.csv"
	print "read dataframes"
	train_df = pd.read_csv(fileNameTrain)
	test_df = pd.read_csv(fileNameTest)
	valid_df = pd.read_csv(fileNameValid)
	

	FlatWeightTrain = train_df.FlatWeight
	FlatWeightTest = test_df.FlatWeight
	FlatWeightValid = valid_df.FlatWeight

	PtWeightTrain = train_df.PtWeight
	PtWeightTest = test_df.PtWeight
	PtWeightValid = valid_df.PtWeight

	feature_list = []
	if variable_type == "all":
		bad_cols = ['jet_pt_response','jet_true_pt','jet_true_eta','jet_true_phi','jet_true_e','jet_respE',
					'jet_respPt','jet_true_m','PtWeight','FlatWeight']
		for col in train_df.columns:
			if str(col) in bad_cols:
				continue
			else:
				feature_list.append(str(col))
	if variable_type == "ftest":
		if jet_type == "LCTopo":
			if etaLow == 0.0:
				feature_list = ['rho','jet_E','jet_pt','jet_eta','jet_DetEta','jet_Wtrk1000','jet_Ntrk1000',
								'jet_ConstitE','jet_nMuSeg','jet_trk_nca','jet_trk_ncasd','jet_trk_rg',
								'jet_trk_c1beta02','jet_EnergyPerSampling_1','jet_EnergyPerSampling_2',
								'jet_EnergyPerSampling_3','jet_EnergyPerSampling_12','jet_EnergyPerSampling_13',
								'rhoEM','jet_Ntrk500','jet_Wtrk500','jet_Ntrk2000','jet_Ntrk3000','jet_Ntrk4000',
								'jet_Wtrk2000','jet_ConstitPt','jet_ConstitEta','abs_phi','abs_ConstitPhi','NPV',
								'actualInteractionsPerCrossing']

			if etaLow == 0.2:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt','jet_eta','jet_DetEta',
								'jet_Wtrk1000','jet_ConstitE','jet_nMuSeg','jet_trk_rg','jet_trk_c1beta02',
								'jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_2',
								'jet_EnergyPerSampling_3','jet_EnergyPerSampling_12','jet_EnergyPerSampling_13',
								'jet_EnergyPerSampling_14','rhoEM','jet_ActiveArea','jet_ActiveArea4vec_pt','jet_Ntrk500','jet_Wtrk500',
								'jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000','jet_ConstitPt','jet_ConstitEta','abs_phi','abs_ConstitPhi']

			if etaLow == 0.7:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt','jet_ConstitE','jet_nMuSeg',
								'jet_trk_rg','jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_2','jet_EnergyPerSampling_3',
								'jet_EnergyPerSampling_12','jet_EnergyPerSampling_13','jet_EnergyPerSampling_15',
								'jet_EnergyPerSampling_16','jet_EnergyPerSampling_17','jet_EnergyPerSampling_18',
								'jet_EnergyPerSampling_19','jet_EnergyPerSampling_20','rhoEM','jet_ActiveArea','jet_ActiveArea4vec_pt','jet_ActiveArea4vec_m',
								'jet_m','jet_Wtrk3000','jet_ConstitPt','jet_ConstitMass','abs_phi','abs_ConstitPhi']
			
			if etaLow == 1.3:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt','jet_Ntrk1000','jet_ConstitE',
								'jet_nMuSeg','jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_c1beta02',
								'jet_EnergyPerSampling_2','jet_EnergyPerSampling_3','jet_EnergyPerSampling_4',
								'jet_EnergyPerSampling_5','jet_EnergyPerSampling_6','jet_EnergyPerSampling_8',
								'jet_EnergyPerSampling_19','jet_EnergyPerSampling_20','rhoEM','jet_ActiveArea','jet_ActiveArea4vec_pt','jet_ActiveArea4vec_m',
								'jet_m','jet_Ntrk500','jet_Ntrk2000','jet_Ntrk3000','jet_Ntrk4000','jet_ConstitPt','jet_ConstitMass']

			if etaLow == 2.0:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt','jet_Ntrk1000','jet_ConstitE',
								'jet_nMuSeg','jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_c1beta02',
								'jet_Wtrk1000','jet_EnergyPerSampling_5','jet_EnergyPerSampling_6',
								'jet_EnergyPerSampling_8','jet_EnergyPerSampling_9','jet_EnergyPerSampling_10',
								'rhoEM','jet_ActiveArea','jet_ActiveArea4vec_pt','jet_ActiveArea4vec_m','jet_Ntrk500','jet_Wtrk500',
								'jet_Ntrk2000','jet_Ntrk3000','jet_Ntrk4000','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000','jet_ConstitPt']

		
		if jet_type == "UFO":
			if etaLow == 0.0:
				feature_list = ['rho','jet_E','jet_pt','jet_Wtrk1000','jet_Ntrk1000',
								'jet_ConstitE','jet_nMuSeg','jet_trk_nca','jet_trk_ncasd','jet_trk_rg',
								'jet_trk_c1beta02','jet_m','jet_Ntrk500',
								'jet_Wtrk500','jet_Ntrk2000','jet_Ntrk3000',
								'jet_Ntrk4000','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000','jet_ConstitPt','jet_ConstitEta',
								'jet_ConstitMass','abs_eta','abs_DetEta','abs_phi','abs_ConstitEta','NPV','abs_ConstitPhi',
								'actualInteractionsPerCrossing']
			if etaLow == 0.2:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt',
								'jet_Wtrk1000','jet_ConstitE','jet_nMuSeg','jet_trk_rg','jet_trk_c1beta02',
								'jet_Ntrk1000','jet_trk_nca','jet_trk_ncasd',
								'jet_m','jet_Ntrk500','jet_Wtrk500',
								'jet_Ntrk2000','jet_Ntrk3000','jet_Ntrk4000','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000',
								'jet_ConstitPt','jet_ConstitEta','jet_ConstitMass','abs_eta','abs_DetEta','abs_phi','abs_ConstitPhi','abs_ConstitEta']
			if etaLow == 0.7:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt','jet_ConstitE','jet_nMuSeg',
								'jet_trk_rg','jet_Wtrk1000','jet_Ntrk1000','jet_trk_nca','jet_trk_ncasd',
								'jet_trk_zg','jet_trk_c1beta02','rhoEM',
								'jet_m','jet_Ntrk500','jet_Wtrk500',
								'jet_Ntrk2000','jet_Ntrk3000','jet_Ntrk4000','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000',
								'jet_ConstitPt','jet_ConstitMass','abs_eta','abs_DetEta','abs_phi','abs_ConstitPhi','abs_ConstitEta']		
			if etaLow == 1.3:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt','jet_Ntrk1000','jet_ConstitE',
								'jet_nMuSeg','jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_c1beta02',
								'abs_eta','jet_Wtrk1000',
								'jet_trk_zg','rhoEM','jet_m',
								'jet_Ntrk500','jet_Wtrk500','jet_Ntrk2000','jet_Ntrk3000','jet_Ntrk4000','jet_Wtrk2000',
								'jet_Wtrk3000','jet_Wtrk4000','jet_ConstitPt','jet_ConstitMass','abs_phi','abs_ConstitPhi']
			if etaLow == 2.0:
				feature_list = ['NPV','actualInteractionsPerCrossing','rho','jet_E','jet_pt','jet_Ntrk1000','jet_ConstitE',
								'jet_nMuSeg','jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_c1beta02',
								'jet_Wtrk1000','jet_phi','rhoEM',
								'jet_m','jet_Ntrk500','jet_Wtrk500',
								'jet_Ntrk2000','jet_Ntrk3000','jet_Ntrk4000','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000',
								'jet_ConstitPt','jet_ConstitPhi','jet_ConstitMass','abs_eta','abs_DetEta','abs_ConstitEta']

	if variable_type == "calorimeter":
		if etaLow == 0.0:
			feature_list = ['jet_E','jet_pt','jet_m','jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_3',
							'jet_EnergyPerSampling_5','jet_EnergyPerSampling_6','jet_EnergyPerSampling_9','jet_EnergyPerSampling_12',
							'jet_EnergyPerSampling_13','jet_EnergyPerSampling_14','jet_EnergyPerSampling_15','jet_EnergyPerSampling_16',
							'jet_EnergyPerSampling_17','jet_EnergyPerSampling_18','jet_EnergyPerSampling_19','jet_EnergyPerSampling_20']
		if etaLow == 0.2:
			feature_list = ['jet_E','jet_pt','jet_m','jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_3',
							'jet_EnergyPerSampling_4','jet_EnergyPerSampling_5','jet_EnergyPerSampling_6','jet_EnergyPerSampling_7',
							'jet_EnergyPerSampling_8','jet_EnergyPerSampling_9','jet_EnergyPerSampling_10','jet_EnergyPerSampling_12',
							'jet_EnergyPerSampling_13','jet_EnergyPerSampling_14','jet_EnergyPerSampling_15','jet_EnergyPerSampling_16',
							'jet_EnergyPerSampling_17','jet_EnergyPerSampling_18','jet_EnergyPerSampling_19','jet_EnergyPerSampling_20']
		if etaLow == 0.7:
			feature_list = ['jet_E','jet_pt','jet_m','jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_3',
							'jet_EnergyPerSampling_4','jet_EnergyPerSampling_5','jet_EnergyPerSampling_6','jet_EnergyPerSampling_7',
							'jet_EnergyPerSampling_8','jet_EnergyPerSampling_9','jet_EnergyPerSampling_10','jet_EnergyPerSampling_11',
							'jet_EnergyPerSampling_12','jet_EnergyPerSampling_13','jet_EnergyPerSampling_14','jet_EnergyPerSampling_15',
							'jet_EnergyPerSampling_16','jet_EnergyPerSampling_17','jet_EnergyPerSampling_18','jet_EnergyPerSampling_19',
							'jet_EnergyPerSampling_20']
		if etaLow == 1.3:
			feature_list = ['jet_E','jet_pt','jet_m','jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_2',
							'jet_EnergyPerSampling_3','jet_EnergyPerSampling_4','jet_EnergyPerSampling_5','jet_EnergyPerSampling_6',
							'jet_EnergyPerSampling_7','jet_EnergyPerSampling_8','jet_EnergyPerSampling_9','jet_EnergyPerSampling_10',
							'jet_EnergyPerSampling_11','jet_EnergyPerSampling_12','jet_EnergyPerSampling_13','jet_EnergyPerSampling_14',
							'jet_EnergyPerSampling_15','jet_EnergyPerSampling_16','jet_EnergyPerSampling_17','jet_EnergyPerSampling_18',
							'jet_EnergyPerSampling_19','jet_EnergyPerSampling_20','jet_EnergyPerSampling_21']
		if etaLow == 2.0:
			feature_list = ['jet_E','jet_pt','jet_m','jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_2',
							'jet_EnergyPerSampling_3','jet_EnergyPerSampling_4','jet_EnergyPerSampling_5','jet_EnergyPerSampling_6',
							'jet_EnergyPerSampling_7','jet_EnergyPerSampling_8','jet_EnergyPerSampling_9','jet_EnergyPerSampling_10',
							'jet_EnergyPerSampling_11','jet_EnergyPerSampling_12','jet_EnergyPerSampling_13','jet_EnergyPerSampling_14',
							'jet_EnergyPerSampling_15','jet_EnergyPerSampling_16','jet_EnergyPerSampling_17','jet_EnergyPerSampling_18',
							'jet_EnergyPerSampling_19','jet_EnergyPerSampling_20','jet_EnergyPerSampling_21','jet_EnergyPerSampling_22',
							'jet_EnergyPerSampling_23']
	if variable_type == "nocalorimeter":
		feature_list = []
		bad_cols = ['jet_pt_response','jet_true_pt','jet_true_eta','jet_true_phi','jet_true_e','jet_respE',
					'jet_respPt','jet_true_m','PtWeight','FlatWeight','jet_EnergyPerSampling_0','jet_EnergyPerSampling_1','jet_EnergyPerSampling_2',
					'jet_EnergyPerSampling_3','jet_EnergyPerSampling_4','jet_EnergyPerSampling_5','jet_EnergyPerSampling_6',
					'jet_EnergyPerSampling_7','jet_EnergyPerSampling_8','jet_EnergyPerSampling_9','jet_EnergyPerSampling_10',
					'jet_EnergyPerSampling_11','jet_EnergyPerSampling_12','jet_EnergyPerSampling_13','jet_EnergyPerSampling_14',
					'jet_EnergyPerSampling_15','jet_EnergyPerSampling_16','jet_EnergyPerSampling_17','jet_EnergyPerSampling_18',
					'jet_EnergyPerSampling_19','jet_EnergyPerSampling_20','jet_EnergyPerSampling_21','jet_EnergyPerSampling_22',
					'jet_EnergyPerSampling_23','jet_EnergyPerSampling_24']
		for col in train_df.columns:
			if str(col) in bad_cols:
				continue
			else:
				feature_list.append(str(col))

	print "Create inputs"
	y_train = train_df.jet_pt_response
	y_test = test_df.jet_pt_response
	y_valid = valid_df.jet_pt_response

	jet_JESPt_train = train_df.jet_JESPt
	jet_JESPt_test = test_df.jet_JESPt
	jet_JESPt_valid = valid_df.jet_JESPt

	jet_true_pt_train = train_df.jet_true_pt
	jet_true_pt_test = test_df.jet_true_pt
	jet_true_pt_valid = valid_df.jet_true_pt
	for col in train_df.columns:
		if str(col) in feature_list:
			continue
		else:
			train_df.drop(str(col),axis=1,inplace=True)
			test_df.drop(str(col),axis=1,inplace=True)
			valid_df.drop(str(col),axis=1,inplace=True)

	X_train = train_df.copy()
	X_train = X_train.values
	y_train = y_train.values

	X_test = test_df.copy()
	X_test = X_test.values
	y_test = y_test.values

	X_valid = valid_df.copy()
	X_valid = X_valid.values
	y_valid = y_valid.values

	train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_20201222.csv'
	test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_20201222.csv'
	valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_20201222.csv'
	if scale_target == "no":
		train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_targetIsNotScaled_20201222.csv'
		test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_targetIsNotScaled_20201222.csv'
		valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_targetIsNotScaled_20201222.csv'
	if variable_type == "calorimeter":
		train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_20201222.csv'
		test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_20201222.csv'
		valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_20201222.csv'
		if scale_target == "no":
			train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_targetIsNotScaled_20201222.csv'
			test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_targetIsNotScaled_20201222.csv'
			valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_targetIsNotScaled_20201222.csv'
	if variable_type == "nocalorimeter":
		train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_20201222.csv'
		test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_20201222.csv'
		valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_20201222.csv'
		if scale_target == "no":
			train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_targetIsNotScaled_20201222.csv'
			test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_targetIsNotScaled_20201222.csv'
			valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_targetIsNotScaled_20201222.csv'

	if variable_type == "all":
		train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_20201222.csv'
		test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_20201222.csv'
		valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_20201222.csv'
		if scale_target == "no":
			train_name = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_targetIsNotScaled_20201222.csv'
			test_name = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_targetIsNotScaled_20201222.csv'
			valid_name = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_targetIsNotScaled_20201222.csv'

	

	print 'train_name =', train_name
	print 'test_name =', test_name
	print 'valid_name =', valid_name

	if jet_type == "LCTopo":
		saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models/'
		if variable_type == "calorimeter":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models_Calorimeter/'
		if variable_type == "nocalorimeter":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models_noCalorimeter/'
		if variable_type == "all":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models_all/'
		if scale_target == "no":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models_targetIsNotScaled/'
			if variable_type == "calorimeter":
				saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models_targetIsNotScaledCalorimeter/'
			if variable_type == "nocalorimeter":
				saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models_targetIsNotScaledNoCalorimeter/'
			if variable_type == "all":
				saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/models_targetIsNotScaledAll/'
				
	else:
		saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models/'
		if variable_type == "calorimeter":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models_Calorimeter/'
		if variable_type == "nocalorimeter":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models_noCalorimeter/'
		if variable_type == "all":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models_all/'
		if scale_target == "no":
			saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models_targetIsNotScaled/'
			if variable_type == "calorimeter":
				saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models_targetIsNotScaledCalorimeter/'
			if variable_type == "nocalorimeter":
				saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models_targetIsNotScaledNoCalorimeter/'
			if variable_type == "all":
				saveFolder = '/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/models_targetIsNotScaledAll/'


	print "load NN model"
	model_olm1 = keras.models.load_model(saveFolder+'OneLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_1/OneLayerModel.h5')
	model_olm2 = keras.models.load_model(saveFolder+'OneLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_2/OneLayerModel.h5')
	model_olm3 = keras.models.load_model(saveFolder+'OneLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_3/OneLayerModel.h5')
	model_tlm1 = keras.models.load_model(saveFolder+'TwoLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_1/TwoLayerModel.h5')
	model_tlm2 = keras.models.load_model(saveFolder+'TwoLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_2/TwoLayerModel.h5')
	model_tlm3 = keras.models.load_model(saveFolder+'TwoLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_3/TwoLayerModel.h5')
	model_thlm1 = keras.models.load_model(saveFolder+'ThreeLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_1/ThreeLayerModel.h5')
	model_thlm2 = keras.models.load_model(saveFolder+'ThreeLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_2/ThreeLayerModel.h5')
	model_thlm3 = keras.models.load_model(saveFolder+'ThreeLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_3/ThreeLayerModel.h5')
	model_flm1 = keras.models.load_model(saveFolder+'FourLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_1/FourLayerModel.h5')
	model_flm2 = keras.models.load_model(saveFolder+'FourLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_2/FourLayerModel.h5')
	model_flm3 = keras.models.load_model(saveFolder+'FourLayerModel/'+'etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'/fold_3/FourLayerModel.h5')


	print "Make predictions OneLayerModel"
	y_train_olm1 = model_olm1.predict(X_train)
	y_test_olm1 = model_olm1.predict(X_test)
	y_valid_olm1 = model_olm1.predict(X_valid)

	y_train_olm2 = model_olm2.predict(X_train)
	y_test_olm2 = model_olm2.predict(X_test)
	y_valid_olm2 = model_olm2.predict(X_valid)

	y_train_olm3 = model_olm3.predict(X_train)
	y_test_olm3 = model_olm3.predict(X_test)
	y_valid_olm3 = model_olm3.predict(X_valid)

	print "Make predictions TwoLayerModel"
	y_train_tlm1 = model_tlm1.predict(X_train)
	y_test_tlm1 = model_tlm1.predict(X_test)
	y_valid_tlm1 = model_tlm1.predict(X_valid)

	y_train_tlm2 = model_tlm2.predict(X_train)
	y_test_tlm2 = model_tlm2.predict(X_test)
	y_valid_tlm2 = model_tlm2.predict(X_valid)

	y_train_tlm3 = model_tlm3.predict(X_train)
	y_test_tlm3 = model_tlm3.predict(X_test)
	y_valid_tlm3 = model_tlm3.predict(X_valid)

	print "Make predictions ThreeLayerModel"
	y_train_thlm1 = model_thlm1.predict(X_train)
	y_test_thlm1 = model_thlm1.predict(X_test)
	y_valid_thlm1 = model_thlm1.predict(X_valid)

	y_train_thlm2 = model_thlm2.predict(X_train)
	y_test_thlm2 = model_thlm2.predict(X_test)
	y_valid_thlm2 = model_thlm2.predict(X_valid)

	y_train_thlm3 = model_thlm3.predict(X_train)
	y_test_thlm3 = model_thlm3.predict(X_test)
	y_valid_thlm3 = model_thlm3.predict(X_valid)

	print "Make predictions FourLayerModel"
	y_train_flm1 = model_flm1.predict(X_train)
	y_test_flm1 = model_flm1.predict(X_test)
	y_valid_flm1 = model_flm1.predict(X_valid)

	y_train_flm2 = model_flm2.predict(X_train)
	y_test_flm2 = model_flm2.predict(X_test)
	y_valid_flm2 = model_flm2.predict(X_valid)

	y_train_flm3 = model_flm3.predict(X_train)
	y_test_flm3 = model_flm3.predict(X_test)
	y_valid_flm3 = model_flm3.predict(X_valid)

	#####################
	y_train_olm1_matrix = y_train_olm1
	y_test_olm1_matrix = y_test_olm1
	y_valid_olm1_matrix = y_valid_olm1

	y_train_olm2_matrix = y_train_olm2
	y_test_olm2_matrix = y_test_olm2
	y_valid_olm2_matrix = y_valid_olm2

	y_train_olm3_matrix = y_train_olm3
	y_test_olm3_matrix = y_test_olm3
	y_valid_olm3_matrix = y_valid_olm3

	####################
	y_train_tlm1_matrix = y_train_tlm1
	y_test_tlm1_matrix = y_test_tlm1
	y_valid_tlm1_matrix = y_valid_tlm1

	y_train_tlm2_matrix = y_train_tlm2
	y_test_tlm2_matrix = y_test_tlm2
	y_valid_tlm2_matrix = y_valid_tlm2

	y_train_tlm3_matrix = y_train_tlm3
	y_test_tlm3_matrix = y_test_tlm3
	y_valid_tlm3_matrix = y_valid_tlm3
	####################
	y_train_thlm1_matrix = y_train_thlm1
	y_test_thlm1_matrix = y_test_thlm1
	y_valid_thlm1_matrix = y_valid_thlm1

	y_train_thlm2_matrix = y_train_thlm2
	y_test_thlm2_matrix = y_test_thlm2
	y_valid_thlm2_matrix = y_valid_thlm2

	y_train_thlm3_matrix = y_train_thlm3
	y_test_thlm3_matrix = y_test_thlm3
	y_valid_thlm3_matrix = y_valid_thlm3
	####################
	####################
	y_train_flm1_matrix = y_train_flm1
	y_test_flm1_matrix = y_test_flm1
	y_valid_flm1_matrix = y_valid_flm1

	y_train_flm2_matrix = y_train_flm2
	y_test_flm2_matrix = y_test_flm2
	y_valid_flm2_matrix = y_valid_flm2

	y_train_flm3_matrix = y_train_flm3
	y_test_flm3_matrix = y_test_flm3
	y_valid_flm3_matrix = y_valid_flm3
	####################

	y_train_olm1 = []
	y_test_olm1 = []
	y_valid_olm1 = []

	y_train_olm2 = []
	y_test_olm2 = []
	y_valid_olm2 = []

	y_train_olm3 = []
	y_test_olm3 = []
	y_valid_olm3 = []

	y_train_tlm1 = []
	y_test_tlm1 = []
	y_valid_tlm1 = []

	y_train_tlm2 = []
	y_test_tlm2 = []
	y_valid_tlm2 = []

	y_train_tlm3 = []
	y_test_tlm3 = []
	y_valid_tlm3 = []

	y_train_thlm1 = []
	y_test_thlm1 = []
	y_valid_thlm1 = []

	y_train_thlm2 = []
	y_test_thlm2 = []
	y_valid_thlm2 = []

	y_train_thlm3 = []
	y_test_thlm3 = []
	y_valid_thlm3 = []



	y_train_flm1 = []
	y_test_flm1 = []
	y_valid_flm1 = []

	y_train_flm2 = []
	y_test_flm2 = []
	y_valid_flm2 = []

	y_train_flm3 = []
	y_test_flm3 = []
	y_valid_flm3 = []

	for i in range(len(y_train_olm1_matrix)):
		y_train_olm1.append(y_train_olm1_matrix[i][0])
		y_train_olm2.append(y_train_olm2_matrix[i][0])
		y_train_olm3.append(y_train_olm3_matrix[i][0])
		y_train_tlm1.append(y_train_tlm1_matrix[i][0])
		y_train_tlm2.append(y_train_tlm2_matrix[i][0])
		y_train_tlm3.append(y_train_tlm3_matrix[i][0])
		y_train_thlm1.append(y_train_thlm1_matrix[i][0])
		y_train_thlm2.append(y_train_thlm2_matrix[i][0])
		y_train_thlm3.append(y_train_thlm3_matrix[i][0])
		y_train_flm1.append(y_train_flm1_matrix[i][0])
		y_train_flm2.append(y_train_flm2_matrix[i][0])
		y_train_flm3.append(y_train_flm3_matrix[i][0])
	for i in range(len(y_test_olm1_matrix)):
		y_test_olm1.append(y_test_olm1_matrix[i][0])
		y_test_olm2.append(y_test_olm2_matrix[i][0])
		y_test_olm3.append(y_test_olm3_matrix[i][0])
		y_test_tlm1.append(y_test_tlm1_matrix[i][0])
		y_test_tlm2.append(y_test_tlm2_matrix[i][0])
		y_test_tlm3.append(y_test_tlm3_matrix[i][0])
		y_test_thlm1.append(y_test_thlm1_matrix[i][0])
		y_test_thlm2.append(y_test_thlm2_matrix[i][0])
		y_test_thlm3.append(y_test_thlm3_matrix[i][0])
		y_test_flm1.append(y_test_flm1_matrix[i][0])
		y_test_flm2.append(y_test_flm2_matrix[i][0])
		y_test_flm3.append(y_test_flm3_matrix[i][0])
	for i in range(len(y_valid_olm1_matrix)):
		y_valid_olm1.append(y_valid_olm1_matrix[i][0])
		y_valid_olm2.append(y_valid_olm2_matrix[i][0])
		y_valid_olm3.append(y_valid_olm3_matrix[i][0])
		y_valid_tlm1.append(y_valid_tlm1_matrix[i][0])
		y_valid_tlm2.append(y_valid_tlm2_matrix[i][0])
		y_valid_tlm3.append(y_valid_tlm3_matrix[i][0])
		y_valid_thlm1.append(y_valid_thlm1_matrix[i][0])
		y_valid_thlm2.append(y_valid_thlm2_matrix[i][0])
		y_valid_thlm3.append(y_valid_thlm3_matrix[i][0])
		y_valid_flm1.append(y_valid_flm1_matrix[i][0])
		y_valid_flm2.append(y_valid_flm2_matrix[i][0])
		y_valid_flm3.append(y_valid_flm3_matrix[i][0])


	pred_train_df = pd.DataFrame()
	pred_test_df = pd.DataFrame()
	pred_valid_df = pd.DataFrame()

	########################
	pred_train_df['jet_pt'] = train_df.jet_pt
	pred_train_df['jet_JESPt'] = jet_JESPt_train
	pred_train_df['jet_true_pt'] = jet_true_pt_train
	pred_train_df['jet_pt_response'] = y_train
	pred_train_df['y_olm1'] = y_train_olm1
	pred_train_df['y_olm2'] = y_train_olm2
	pred_train_df['y_olm3'] = y_train_olm3
	#pred_train_df['y_olm'] = (y_train_olm1 + y_train_olm2 + y_train_olm3) / 3
	pred_train_df['y_tlm1'] = y_train_tlm1
	pred_train_df['y_tlm2'] = y_train_tlm2
	pred_train_df['y_tlm3'] = y_train_tlm3
	#pred_train_df['y_tlm'] = (y_train_tlm1 + y_train_tlm2 + y_train_tlm3) / 3
	pred_train_df['y_thlm1'] = y_train_thlm1
	pred_train_df['y_thlm2'] = y_train_thlm2
	pred_train_df['y_thlm3'] = y_train_thlm3
	#pred_train_df['y_thlm'] = (y_train_thlm1 + y_train_thlm2 + y_train_thlm3) / 3
	pred_train_df['y_flm1'] = y_train_thlm1
	pred_train_df['y_flm2'] = y_train_thlm2
	pred_train_df['y_flm3'] = y_train_thlm3
	#pred_train_df['y_flm'] = (y_train_flm1 + y_train_flm2 + y_train_flm3) / 3
	pred_train_df['PtWeight'] = PtWeightTrain.values
	pred_train_df['FlatWeight'] = FlatWeightTrain.values

	pred_test_df['jet_pt'] = test_df.jet_pt
	pred_test_df['jet_JESPt'] = jet_JESPt_test
	pred_test_df['jet_true_pt'] = jet_true_pt_test
	pred_test_df['jet_pt_response'] = y_test
	pred_test_df['y_olm1'] = y_test_olm1
	pred_test_df['y_olm2'] = y_test_olm2
	pred_test_df['y_olm3'] = y_test_olm3
	#pred_test_df['y_olm'] = (y_test_olm1 + y_test_olm2 + y_test_olm3) / 3
	pred_test_df['y_tlm1'] = y_test_tlm1
	pred_test_df['y_tlm2'] = y_test_tlm2
	pred_test_df['y_tlm3'] = y_test_tlm3
	#pred_test_df['y_tlm'] = (y_test_tlm1 + y_test_tlm2 + y_test_tlm3) / 3
	pred_test_df['y_thlm1'] = y_test_thlm1
	pred_test_df['y_thlm2'] = y_test_thlm2
	pred_test_df['y_thlm3'] = y_test_thlm3
	#pred_test_df['y_thlm'] = (y_test_thlm1 + y_test_thlm2 + y_test_thlm3) / 3
	pred_test_df['y_flm1'] = y_test_flm1
	pred_test_df['y_flm2'] = y_test_flm2
	pred_test_df['y_flm3'] = y_test_flm3
	#pred_test_df['y_flm'] = (y_test_flm1 + y_test_flm2 + y_test_flm3) / 3
	pred_test_df['PtWeight'] = PtWeightTest.values
	pred_test_df['FlatWeight'] = FlatWeightTest.values

	pred_valid_df['jet_pt'] = valid_df.jet_pt
	pred_valid_df['jet_JESPt'] = jet_JESPt_valid
	pred_valid_df['jet_true_pt'] = jet_true_pt_valid 
	pred_valid_df['jet_pt_response'] = y_valid
	pred_valid_df['y_olm1'] = y_valid_olm1
	pred_valid_df['y_olm2'] = y_valid_olm2
	pred_valid_df['y_olm3'] = y_valid_olm3
	#pred_valid_df['y_olm'] = (y_valid_olm1 + y_valid_olm2 + y_valid_olm3) / 3
	pred_valid_df['y_tlm1'] = y_valid_tlm1
	pred_valid_df['y_tlm2'] = y_valid_tlm2
	pred_valid_df['y_tlm3'] = y_valid_tlm3
	#pred_valid_df['y_tlm'] = (y_valid_tlm1 + y_valid_tlm2 + y_valid_tlm3) / 3
	pred_valid_df['y_thlm1'] = y_valid_thlm1
	pred_valid_df['y_thlm2'] = y_valid_thlm2
	pred_valid_df['y_thlm3'] = y_valid_thlm3
	#pred_valid_df['y_thlm'] = (y_valid_thlm1 + y_valid_thlm2 + y_valid_thlm3) / 3
	pred_valid_df['y_flm1'] = y_valid_flm1
	pred_valid_df['y_flm2'] = y_valid_flm2
	pred_valid_df['y_flm3'] = y_valid_flm3
	#pred_valid_df['y_flm'] = (y_valid_flm1 + y_valid_flm2 + y_valid_flm3) / 3
	pred_valid_df['PtWeight'] = PtWeightValid.values
	pred_valid_df['FlatWeight'] = FlatWeightValid.values


	print train_df.shape, test_df.shape, valid_df.shape
	print pred_train_df.shape, pred_test_df.shape, pred_valid_df.shape

	print "Create csv files"
	pred_train_df.to_csv(train_name,index=False)
	print "scaled_df_pred_train is created"
	pred_test_df.to_csv(test_name,index=False)
	print "scaled_df_pred_test is created"
	pred_valid_df.to_csv(valid_name,index=False)
	print "scaled_df_pred_valid is created"
	print "Done"


parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
parser.add_option('--scale_target', default = "yes", type = "string", help = "scale target variable (yes or no)")
parser.add_option('--variable_type', default = "ftest", type = "string", help = "type of varible selection")
(opt, args) = parser.parse_args()
MakeNNPredictions(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh,scale_target=opt.scale_target,variable_type=opt.variable_type)
