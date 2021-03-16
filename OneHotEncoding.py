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
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 999)

train_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/train_df_eta02_20200312.csv")
test_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/test_df_eta02_20200312.csv")
valid_df = pd.read_csv("/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/valid_df_eta02_20200312.csv")

df_array = [train_df,test_df,valid_df]
dataFrame = df_array[0]
for index in range(1,len(df_array)):
	print "index =", index
	dataFrame = pd.concat([dataFrame, df_array[index]], axis=0, ignore_index=True)

# Label Encoders
le_nca = LabelEncoder()
dataFrame['jet_trk_nca_le'] = le_nca.fit_transform(dataFrame[['jet_trk_nca']])
train_df['jet_trk_nca_le'] = le_nca.transform(train_df[['jet_trk_nca']])
test_df['jet_trk_nca_le'] = le_nca.transform(test_df[['jet_trk_nca']])
valid_df['jet_trk_nca_le'] = le_nca.transform(valid_df[['jet_trk_nca']])

le_ncasd = LabelEncoder()
dataFrame['jet_trk_ncasd_le'] = le_ncasd.fit_transform(dataFrame[['jet_trk_ncasd']])
train_df['jet_trk_ncasd_le'] = le_ncasd.transform(train_df[['jet_trk_ncasd']])
test_df['jet_trk_ncasd_le'] = le_ncasd.transform(test_df[['jet_trk_ncasd']])
valid_df['jet_trk_ncasd_le'] = le_ncasd.transform(valid_df[['jet_trk_ncasd']])


# encode nca
ohe_nca = OneHotEncoder(handle_unknown='ignore')

enc_nca_df = pd.DataFrame(ohe_nca.fit_transform(dataFrame[['jet_trk_nca_le']]).toarray())
enc_nca_df.columns = ohe_nca.get_feature_names(['jet_trk_nca_le'])
dataFrame = dataFrame.join(enc_nca_df)

enc_nca_train = pd.DataFrame(ohe_nca.transform(train_df[['jet_trk_nca_le']]).toarray())
enc_nca_train.columns = ohe_nca.get_feature_names(['jet_trk_nca_le'])
train_df = train_df.join(enc_nca_train)

enc_nca_test = pd.DataFrame(ohe_nca.transform(test_df[['jet_trk_nca_le']]).toarray())
enc_nca_test.columns = ohe_nca.get_feature_names(['jet_trk_nca_le'])
test_df = test_df.join(enc_nca_test)

enc_nca_valid = pd.DataFrame(ohe_nca.transform(valid_df[['jet_trk_nca_le']]).toarray())
enc_nca_valid.columns = ohe_nca.get_feature_names(['jet_trk_nca_le'])
valid_df = valid_df.join(enc_nca_valid)
'''
# encode ncasd
ohe_ncasd = OneHotEncoder(handle_unknown='ignore')

enc_ncasd_df = pd.DataFrame(ohe_ncasd.fit_transform(dataFrame[['jet_trk_ncasd_le']]).toarray())
enc_ncasd_df.columns = ohe_ncasd.get_feature_names(['jet_trk_ncasd_le'])
dataFrame = dataFrame.join(enc_ncasd_df)


enc_ncasd_train = pd.DataFrame(ohe_ncasd.transform(train_df[['jet_trk_ncasd_le']]).toarray())
enc_ncasd_train.columns = ohe_ncasd.get_feature_names(['jet_trk_ncasd_le'])
train_df = train_df.join(enc_ncasd_train)

enc_ncasd_test = pd.DataFrame(ohe_ncasd.transform(test_df[['jet_trk_ncasd_le']]).toarray())
enc_ncasd_test.columns = ohe_ncasd.get_feature_names(['jet_trk_ncasd_le'])
test_df = test_df.join(enc_ncasd_test)

enc_ncasd_valid = pd.DataFrame(ohe_ncasd.transform(valid_df[['jet_trk_ncasd_le']]).toarray())
enc_ncasd_valid.columns = ohe_ncasd.get_feature_names(['jet_trk_ncasd_le'])
valid_df = valid_df.join(enc_ncasd_valid)
'''
print 'Train'
print train_df.columns

print 'Test'
print test_df.columns


print 'Valid'
print valid_df.columns
