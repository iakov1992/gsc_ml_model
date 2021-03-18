import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import keras.backend as K
from NeuralNetworkConfiguration import *
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


print 'Select Model: (0)OneLayerModel,(1)TwoLayerModel,(2)OneLayerModelDrawers,(3)ConvolutionalModel'
select_model = int(input())
model_list = ['OneLayerModel','TwoLayerModel','OneLayerModelDrawers','ConvolutionalModel']
if select_model >= len(model_list):
	print 'Wrong input, select again:'
	select_model = int(input())

drawers = False
use_additional_features = False

selected_model = model_list[select_model]
print "Selected Model:", selected_model

print "Read the dataframe"
train_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_train_df_eta02_20200520.csv")

FlatWeight = train_df.FlatWeight
X = train_df.drop(['jet_true_pt','FlatWeight','PtWeight'],axis=1)
y = train_df.jet_true_pt


if selected_model == "OneLayerModel":
	neurons = int(input('Select the number of neurons for OneLayerModel (one number):'))
	X_nn,y_nn,model,save_model_name,drawers = getOneLayerNN(X,y,neurons,use_additional_features)
elif selected_model == "TwoLayerModel":
	neuron1 = int(input('Select the number of neurons for TwoLayerModel (first layer):'))
	neuron2 = int(input('Select the number of neurons for TwoLayerModel (second layer):'))
	X_nn,y_nn,model,save_model_name,drawers = getTwoLayerNN(X,y,neuron1,neuron2,use_additional_features)

elif selected_model == "OneLayerModelDrawers":
	neurons = int(input('Select the number of neurons for OneLayerModel (one number):'))
	X_nn,y_nn,y_class_nn,class_weight,model,save_model_name,drawers = getOneLayerNNDrawers(X,y,neurons,use_additional_features)

elif selected_model == "ConvolutionalModel":
	neuron1 = int(input('Select the number of neurons for ConvolutionalModel (first dense layer):'))
	neuron_conv1 = int(input('Select the number of neurons for ConvolutionalModel (first convolutional layer):'))
	neuron_conv2 = int(input('Select the number of neurons for ConvolutionalModel (second convolutional layer):'))
	neuron2 = int(input('Select the number of neurons for ConvolutionalModel (second dense layer):'))
	X_nn,y_nn,model,save_model_name,drawers = getConvolutionalModel(X,y,neuron1,neuron_conv1,neuron_conv2,neuron2,use_additional_features)
print 'Select Model: (0)OneLayerModel,(1)TwoLayerModel,(2)OneLayerModelDrawers,(3)ConvolutionalModel'
select_model = int(input())
model_list = ['OneLayerModel','TwoLayerModel','OneLayerModelDrawers','ConvolutionalModel']
if select_model >= len(model_list):
	print 'Wrong input, select again:'
	select_model = int(input())

drawers = False
use_additional_features = False

selected_model = model_list[select_model]
print "Selected Model:", selected_model

print "Read the dataframe"
train_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_train_df_eta02_20200520.csv")

FlatWeight = train_df.FlatWeight
X = train_df.drop(['jet_true_pt','FlatWeight','PtWeight'],axis=1)
y = train_df.jet_true_pt


if selected_model == "OneLayerModel":
	neurons = int(input('Select the number of neurons for OneLayerModel (one number):'))
	X_nn,y_nn,model,save_model_name,drawers = getOneLayerNN(X,y,neurons,use_additional_features)
elif selected_model == "TwoLayerModel":
	neuron1 = int(input('Select the number of neurons for TwoLayerModel (first layer):'))
	neuron2 = int(input('Select the number of neurons for TwoLayerModel (second layer):'))
	X_nn,y_nn,model,save_model_name,drawers = getTwoLayerNN(X,y,neuron1,neuron2,use_additional_features)

elif selected_model == "OneLayerModelDrawers":
	neurons = int(input('Select the number of neurons for OneLayerModel (one number):'))
	X_nn,y_nn,y_class_nn,class_weight,model,save_model_name,drawers = getOneLayerNNDrawers(X,y,neurons,use_additional_features)

elif selected_model == "ConvolutionalModel":
	neuron1 = int(input('Select the number of neurons for ConvolutionalModel (first dense layer):'))
	neuron_conv1 = int(input('Select the number of neurons for ConvolutionalModel (first convolutional layer):'))
	neuron_conv2 = int(input('Select the number of neurons for ConvolutionalModel (second convolutional layer):'))
	neuron2 = int(input('Select the number of neurons for ConvolutionalModel (second dense layer):'))
	X_nn,y_nn,model,save_model_name,drawers = getConvolutionalModel(X,y,neuron1,neuron_conv1,neuron_conv2,neuron2,use_additional_features)



model.compile(loss = {'regression_output': keras.losses.mape},
              optimizer = keras.optimizers.Adam(lr=1.e-4),
              metrics = {'regression_output': [keras.metrics.mape,"mse"]})
print model.summary()

