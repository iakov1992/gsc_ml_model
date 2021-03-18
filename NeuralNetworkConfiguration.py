import keras 
from keras.models import Model
from keras.layers import *
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import h5py
import warnings
import keras.backend as K
warnings.filterwarnings("ignore")


def getOneLayerNN(input_shape):
	model_name = "OneLayerModel"
	save_model_name = "OneLayerModel.h5"
	inputs = Input(shape = (input_shape,),name='input_layer')
	dense = Dense(300,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer1')(inputs)
	regression_output = Dense(1,activation='linear',name='regression_output')(dense)
	model = Model(inputs,[regression_output],name=model_name) 
	return model,save_model_name

def getTwoLayerNN(input_shape):
	print ("TwoLayerModel is selected")
	model_name = "TwoLayerModel"
	save_model_name = "TwoLayerModel.h5"
	inputs = Input(shape = (input_shape,),name='input_layer')
	dense0 = Dense(60,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer1')(inputs)
	dense1 = Dense(40,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer2')(dense0)
	regression_output = Dense(1,activation='linear',name='regression_output')(dense1)
	model = Model(inputs,[regression_output],name=model_name) 
	return model,save_model_name

def getThreeLayerNN(input_shape):
	print ("ThreeLayerModel is selected")
	model_name = "ThreeLayerModel"
	save_model_name = "ThreeLayerModel.h5"
	inputs = Input(shape = (input_shape,),name='input_layer')
	dense0 = Dense(512,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer1')(inputs)
	dense1 = Dense(256,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer2')(dense0)
	dense2 = Dense(128,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer3')(dense1)
	regression_output = Dense(1,activation='linear',name='regression_output')(dense2)
	model = Model(inputs,[regression_output],name=model_name) 
	return model,save_model_name

def getFourLayerNN(input_shape):
	print ("FourLayerModel is selected")
	model_name = "FourLayerModel"
	save_model_name = "FourLayerModel.h5"
	inputs = Input(shape = (input_shape,),name='input_layer')
	dense0 = Dense(1024,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer1')(inputs)
	dense1 = Dense(512,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer2')(dense0)
	dense2 = Dense(256,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer3')(dense1)
	dense3 = Dense(128,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer4')(dense2)
	regression_output = Dense(1,activation='linear',name='regression_output')(dense3)
	model = Model(inputs,[regression_output],name=model_name) 
	return model,save_model_name

def getFiveLayerNN(input_shape):
	print ("FiveLayerModel is selected")
	model_name = "FiveLayerModel"
	save_model_name = "FiveLayerModel.h5"
	inputs = Input(shape = (input_shape,),name='input_layer')
	dense0 = Dense(1024,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer1')(inputs)
	dropout0 = Dropout(0.2)(dense0)
	dense1 = Dense(512,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer2')(dropout0)
	dropout1 = Dropout(0.2)(dense1)
	dense2 = Dense(256,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer3')(dropout1)
	dropout2 = Dropout(0.2)(dense2)
	dense3 = Dense(128,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer4')(dropout2)
	dense4 = Dense(64,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer5')(dense3)
	regression_output = Dense(1,activation='linear',name='regression_output')(dense4)
	model = Model(inputs,[regression_output],name=model_name) 
	return model,save_model_name

def getFiveLayerNoDropoutNN(input_shape):
	print ("FiveLayerModel without Dropout is selected")
	model_name = "FiveLayerModelNoDropout"
	save_model_name = "FiveLayerModelNoDropout.h5"
	inputs = Input(shape = (input_shape,),name='input_layer')
	dense0 = Dense(1024,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer1')(inputs)
	dense1 = Dense(512,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer2')(dense0)
	dense2 = Dense(256,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer3')(dense1)
	dense3 = Dense(128,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer4')(dense2)
	dense4 = Dense(64,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer5')(dense3)
	regression_output = Dense(1,activation='linear',name='regression_output')(dense4)
	model = Model(inputs,[regression_output],name=model_name) 
	return model,save_model_name

def getConvolutionalModel(input_shape):
	print ("ConvolutionalModel is selected")
	model_name = "ConvolutionalModel"
	save_model_name = "ConvolutionalModel.h5"
	inputs = Input(shape = (input_shape,),name='input_layer')
	dense1 = Dense(1024,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer1')(inputs)
	dense2 = Dense(512,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer2')(dense1)
	reshape1 = Reshape((512,1),name='reshape')(dense2)
	conv1 = Conv1D(256, 2, activation="relu",name='conv1')(reshape1)
	maxpooling1 = MaxPooling1D(pool_size=2)(conv1)
	conv2 = Conv1D(128, 2, activation="relu",name='conv2')(maxpooling1)
	maxpooling2 = MaxPooling1D(pool_size=2)(conv2)
	conv3 = Conv1D(64, 2, activation="relu",name='conv3')(maxpooling2)
	flatten1 = Flatten()(conv3)
	dense3 = Dense(128,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer3')(flatten1)
	dense4 = Dense(64,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer4')(dense3)
	dense5 = Dense(32,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer5')(dense4)
	dense6 = Dense(16,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer6')(dense5)
	reshape2 = Reshape((16,1),name='reshape2')(dense6)
	conv4 = Conv1D(16, 2, activation="relu",name='conv4')(reshape2)
	maxpooling3 = MaxPooling1D(pool_size=2)(conv4)
	conv5 = Conv1D(8, 2, activation="relu",name='conv5')(maxpooling3)
	flatten2 = Flatten()(conv5)
	dense7 = Dense(8,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer7')(flatten2)
	dense8 = Dense(4,activation='relu',kernel_initializer=keras.initializers.he_uniform(),name='layer8')(dense7)
	regression_output = Dense(1,activation='linear',name='regression_output')(dense8)
	model = Model(inputs,[regression_output],name=model_name) 
	return model,save_model_name
