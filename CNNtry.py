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
warnings.filterwarnings("ignore")
'''
def lgk(y_true, y_pred):
  h = 1e-1
  alpha=1e-6
  return -np.exp(-0.5*(y_true - y_pred)**2 / h**2) / np.sqrt(2*np.pi) / h + alpha*np.abs(y_true-y_pred)
'''
def lgk(y_true, y_pred):
  h = 1e-1
  alpha=1e-6
  return -K.exp(-0.5*(y_true - y_pred)**2 / h**2) / np.sqrt(2*np.pi) / h + alpha*K.abs(y_true-y_pred)

def relative_lgk(y_true, y_pred):
  h = 1e-1
  alpha=1e-6
  return -np.exp(-0.5*np.divide((y_pred-y_true), y_true)**2 / h**2) / np.sqrt(2*np.pi) / h + alpha*np.abs(np.divide((y_pred-y_true), y_true))



train_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/scaled_train_df_eta02_20200520.csv")

FlatWeight = train_df.FlatWeight
X = train_df.drop(['jet_true_pt','FlatWeight','PtWeight'],axis=1)
'''
X = train_df.drop(['jet_true_pt','FlatWeight','PtWeight','jet_trk_nca_le_0','jet_trk_nca_le_1','jet_trk_nca_le_2',
													'jet_trk_nca_le_3','jet_trk_nca_le_4','jet_trk_nca_le_5','jet_trk_nca_le_6','jet_trk_nca_le_7',
													'jet_trk_nca_le_8','jet_trk_nca_le_9','jet_trk_nca_le_10','jet_trk_nca_le_11','jet_trk_nca_le_12',
													'jet_trk_nca_le_13','jet_trk_nca_le_14','jet_trk_ncasd_le_0','jet_trk_ncasd_le_1','jet_trk_ncasd_le_2',
													'jet_trk_ncasd_le_3','jet_trk_ncasd_le_4','jet_trk_ncasd_le_5','jet_trk_ncasd_le_6','jet_trk_ncasd_le_7',
													'jet_trk_ncasd_le_8','jet_trk_ncasd_le_9','jet_trk_ncasd_le_10','jet_trk_ncasd_le_11','jet_trk_ncasd_le_12',
													'jet_trk_ncasd_le_13','jet_trk_ncasd_le_14'],axis=1)
'''
y = train_df.jet_true_pt

for col in train_df.columns:
	print col

X_nn = X.values
y_nn = y.values
input_dim = len(X.columns)

'''
# 0.00166
model = Sequential()
model.add(Conv1D(128, 2, activation="relu", input_shape=(X_nn.shape[1], 1)))
model.add(Flatten())
model.add(Dense(300, kernel_initializer=keras.initializers.he_normal(seed=17)))
model.add(Activation('relu')) 
model.add(Dense(1))
model.add(Activation('linear')) 

model.compile(optimizer=keras.optimizers.Adam(lr=1.e-4),
              loss=keras.losses.mape,
              metrics= [keras.metrics.mape,"mse"])
'''

'''
# 0.00164
model = Sequential()
model.add(Dense(300, input_dim=input_dim, kernel_initializer=keras.initializers.he_normal(seed=17)))
model.add(Reshape((300,1)))
model.add(Conv1D(128, 2, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 2, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(40, kernel_initializer=keras.initializers.he_normal(seed=17)))
model.add(Activation('relu')) 
model.add(Dense(1))
model.add(Activation('linear')) 
'''


model = Sequential()
model.add(Dense(512, input_dim=input_dim, kernel_initializer=keras.initializers.he_normal(seed=17)))
#model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dense(256, input_dim=input_dim, kernel_initializer=keras.initializers.he_normal(seed=17)))
#model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Reshape((256,1)))
model.add(Conv1D(256, 2, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, 2, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, kernel_initializer=keras.initializers.he_normal(seed=17)))
#model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dense(64, kernel_initializer=keras.initializers.he_normal(seed=17)))
#model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Reshape((64,1)))
model.add(Conv1D(64, 2, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, 2, activation="relu"))
model.add(Flatten())
model.add(Dense(32, kernel_initializer=keras.initializers.he_normal(seed=17)))
#model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dense(16, kernel_initializer=keras.initializers.he_normal(seed=17)))
#model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dense(1, activation="linear"))

model.compile(optimizer=keras.optimizers.Adam(lr=1.e-4),
              loss=keras.losses.mape,
              metrics= [keras.metrics.mape,"mse"])

print model.summary()

model.fit(X_nn,y_nn,epochs=1000000,validation_split=0.3,
          batch_size=128, shuffle='batch', sample_weight=FlatWeight.values,
          callbacks = [EarlyStopping(monitor='val_loss',mode='min', patience = 50, verbose = 1),
                       ModelCheckpoint(filepath='2020_04_20/nn_conv.h5', monitor='val_loss', 
                       save_best_only=True, verbose=1)])


