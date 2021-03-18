import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import BatchNormalization
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
import keras
from NeuralNetworkConfiguration import *
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from optparse import OptionParser





warnings.filterwarnings("ignore")
'''
def lgk(y_true, y_pred):
  h = 1e-1
  alpha=1e-6
  return -np.exp(-0.5*(y_true - y_pred)**2 / h**2) / np.sqrt(2*np.pi) / h + alpha*np.abs(y_true-y_pred)
'''
def mape(y_true,y_pred,sample_weight):
	err = []
	for i in range(len(y_true)):
		sample_error = np.abs(sample_weight[i]*(y_true[i] - y_pred[i])/y_true[i])
		err.append(sample_error)
	return 100 * np.mean(err)


def RunNeuralNetwork(jet_type,etaLow,etaHigh,model,scale_target,variable_type):
	os.environ["MKL_NUM_THREADS"] = "8"
	os.environ["OMP_NUM_THREADS"] = "8"
	print ("Read the dataframe")
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
		
	fileNameTrain = folder_name + "scaled_train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv"
	if scale_target == "no":
		fileNameTrain = folder_name + "scaled_train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_targetIsNotScaled_noNan.csv"
	train_df = pd.read_csv(fileNameTrain)
	FlatWeight = train_df.FlatWeight
	

	'''
	Index([u'jet_phi', u'jet_eta', u'jet_DetEta', u'jet_EnergyPerSampling_0',
       u'jet_EnergyPerSampling_1', u'jet_EnergyPerSampling_2',
       u'jet_EnergyPerSampling_3', u'jet_EnergyPerSampling_5',
       u'jet_EnergyPerSampling_6', u'jet_EnergyPerSampling_9',
       u'jet_EnergyPerSampling_12', u'jet_EnergyPerSampling_13',
       u'jet_EnergyPerSampling_14', u'jet_EnergyPerSampling_15',
       u'jet_EnergyPerSampling_16', u'jet_EnergyPerSampling_17',
       u'jet_EnergyPerSampling_18', u'jet_EnergyPerSampling_19',
       u'jet_EnergyPerSampling_20', u'jet_m', u'jet_ConstitEta',
       u'jet_ConstitPhi', u'NPV', u'actualInteractionsPerCrossing', u'rho',
       u'jet_E', u'jet_pt', u'jet_Wtrk1000', u'jet_Ntrk1000', u'jet_ConstitE',
       u'jet_nMuSeg', u'jet_trk_nca', u'jet_trk_ncasd', u'jet_trk_rg',
       u'jet_trk_zg', u'jet_trk_c1beta02', u'rhoEM', u'jet_Ntrk500',
       u'jet_Wtrk500', u'jet_Ntrk2000', u'jet_Ntrk3000', u'jet_Ntrk4000',
       u'jet_Wtrk2000', u'jet_Wtrk3000', u'jet_Wtrk4000', u'jet_ConstitPt',
       u'jet_ConstitMass'],

	'''
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

	y = train_df.jet_pt_response
	for col in train_df.columns:
		if str(col) in feature_list:
			continue
		else:
			train_df.drop(str(col),axis=1,inplace=True)
	X = train_df.copy()

	print (X.columns)
	print ("min(y) =",y.min())
	print ("DataFrame shape:",X.shape)

	X_nn = X.values
	y_nn = y.values
	input_shape = len(X.columns)

	selected_model = str(model)
	my_patience = 20
	BatchSize = 256	
	if selected_model == "ConvolutionalModel":
		BatchSize = 2048
		my_patience = 20



	with tf.Session(config=tf.ConfigProto(
                	intra_op_parallelism_threads=8,
					inter_op_parallelism_threads=8,
					#allow_soft_placement=True,
					device_count={'CPU':8,'GPU' : 0})) as sess:
		K.set_session(sess)
		# Define the K-fold Cross Validator
		num_folds = 3
		kfold = KFold(n_splits=num_folds, shuffle=True)
		fold_no = 1
		for train, test in kfold.split(X_nn,y_nn):
			print ("Fold #",fold_no)
			if selected_model == "OneLayerModel":
				model,save_model_name = getOneLayerNN(input_shape)
			elif selected_model == "TwoLayerModel":
				model,save_model_name = getTwoLayerNN(input_shape)
			elif selected_model == "ThreeLayerModel":
				model,save_model_name = getThreeLayerNN(input_shape)
			elif selected_model == "FourLayerModel":
				model,save_model_name = getFourLayerNN(input_shape)
			elif selected_model == "FiveLayerModel":
				model,save_model_name = getFiveLayerNN(input_shape)
			elif selected_model == "FiveLayerModelNoDropout":
				model,save_model_name = getFiveLayerNoDropoutNN(input_shape)
			elif selected_model == "ConvolutionalModel":
				model,save_model_name = getConvolutionalModel(input_shape)
			else:
				print("There is no such type of model")
			saveFolder = "jets"
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
			if fold_no == 1:
				if not os.path.exists(saveFolder+str(selected_model)):
					#print("Removing directory",saveFolder+str(selected_model))
					#os.rmdir(saveFolder+str(selected_model))
					print("Create directory:",saveFolder+str(selected_model))
					os.makedirs(saveFolder+str(selected_model))
				if os.path.exists(saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh)):
					print("Removing directory",saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh))
					os.rmdir(saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh))
				print("Create directory:",saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh))
				os.makedirs(saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh))
			print("Create directory:",saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh)+'/fold_'+str(fold_no))
			os.makedirs(saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh)+'/fold_'+str(fold_no))
			file_name = saveFolder+str(selected_model) + '/etaLow_'+str(etaLow)+"_etaHigh_"+str(etaHigh)+'/fold_'+str(fold_no)+"/"+save_model_name
			model.compile(loss = {'regression_output': keras.losses.mape},
					   	  optimizer = keras.optimizers.Adam(lr=1.e-3,clipnorm=1.0),
					   	  metrics = {'regression_output': [keras.metrics.mape,"mse"]})

			print (model.summary())
			model.fit(X_nn[train],{'regression_output':y_nn[train]},
				  	  epochs=100000, validation_split=0.2, batch_size=BatchSize, 
				  	  sample_weight={'regression_output':FlatWeight.values[train]},
				  	  callbacks = [EarlyStopping(monitor='val_loss',mode='min', 
												 patience = my_patience, verbose = 1),
								   ModelCheckpoint(filepath=file_name,
												   monitor='val_loss',save_best_only=True, verbose=1)])
			model_nn = keras.models.load_model(file_name)
			test_prediction = model_nn.predict(X_nn[test])
			print ("Fold #", fold_no,"mape:",mape(y_nn[test],test_prediction,FlatWeight.values[test]))
			fold_no += 1


parser = OptionParser()
parser.add_option('--model', type="string", default = "OneLayerModel")
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
parser.add_option('--scale_target', default = "yes", type = "string", help = "scale target variable (yes or no)")
parser.add_option('--variable_type', default = "ftest", type = "string", help = "type of varible selection")
(opt, args) = parser.parse_args()
RunNeuralNetwork(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh,model=opt.model,scale_target=opt.scale_target,variable_type=opt.variable_type)
