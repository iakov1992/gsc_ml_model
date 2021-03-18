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
from sklearn.neighbors import KNeighborsRegressor

def FillNans(jet_type,etaLow,etaHigh):
    folder_name = "jets"
    if jet_type == "LCTopo":
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
	
    print folder_name
    
    trainName = folder_name + 'train_etaLow'+str(etaLow)+'_etaHigh'+str(etaHigh)+'.csv'
    print trainName
    
    train_df = pd.read_csv(trainName)
    FlatWeight = train_df['FlatWeight']
    PtWeight = train_df['PtWeight']
    train_df['jet_pt_response'] = train_df.jet_pt / train_df.jet_true_pt
    train_df.drop(['PtWeight', 'FlatWeight','jet_true_eta','jet_true_phi','jet_true_e',
                   'jet_respE', 'jet_respPt','jet_true_m', 'jet_JESE','jet_JESPt','jet_JESEta',
                   'jet_JESPhi','jet_JESMass','jet_true_pt','jet_pt_response'],axis=1,inplace=True)
                   
    print "fraction of missing values in jet_trk_nca in the training sample:", float(len(train_df[train_df["jet_trk_nca"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_ncasd in the training sample:", float(len(train_df[train_df["jet_trk_ncasd"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_rg in the training sample:", float(len(train_df[train_df["jet_trk_rg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_zg in the training sample:", float(len(train_df[train_df["jet_trk_zg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_c1beta02 in the training sample:", float(len(train_df[train_df["jet_trk_c1beta02"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_nMuSeg in the training sample:", float(len(train_df[train_df["jet_nMuSeg"]<0.]))/float(len(train_df))
    print "fraction of missing values in NPV in the training sample:", float(len(train_df[train_df["NPV"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk1000 in the training sample:", float(len(train_df[train_df["jet_Wtrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk1000 in the training sample:", float(len(train_df[train_df["jet_Ntrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk500 in the training sample:", float(len(train_df[train_df["jet_Wtrk500"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk500 in the training sample:", float(len(train_df[train_df["jet_Ntrk500"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk2000 in the training sample:", float(len(train_df[train_df["jet_Wtrk2000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk2000 in the training sample:", float(len(train_df[train_df["jet_Ntrk2000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk3000 in the training sample:", float(len(train_df[train_df["jet_Wtrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk3000 in the training sample:", float(len(train_df[train_df["jet_Ntrk3000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk4000 in the training sample:", float(len(train_df[train_df["jet_Wtrk4000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk4000 in the training sample:", float(len(train_df[train_df["jet_Ntrk4000"]<0.]))/float(len(train_df))
    

    df_train = train_df[(train_df['jet_trk_nca'] > 0) & (train_df['jet_trk_ncasd'] > 0) & (train_df['jet_trk_rg'] > 0) &
                        (train_df['jet_trk_zg'] > 0) & (train_df['jet_trk_c1beta02'] > 0) & (train_df['jet_Wtrk1000'] > 0) &
                        (train_df['jet_Wtrk500'] > 0) & (train_df['jet_Wtrk2000'] > 0) & (train_df['jet_Wtrk3000'] > 0) & 
                        (train_df['jet_Wtrk4000'] > 0)]
    
    df_test_0 = train_df[(train_df['jet_trk_nca'] < 0)]
    df_test_1 = train_df[(train_df['jet_trk_ncasd'] < 0)]
    df_test_2 = train_df[(train_df['jet_trk_rg'] < 0)]
    df_test_3 = train_df[(train_df['jet_trk_zg'] < 0)]
    df_test_4 = train_df[(train_df['jet_trk_c1beta02'] < 0)]
    df_test_5 = train_df[(train_df['jet_Wtrk1000'] < 0)]
    df_test_6 = train_df[(train_df['jet_Wtrk500'] < 0)]
    df_test_7 = train_df[(train_df['jet_Wtrk2000'] < 0)]
    df_test_8 = train_df[(train_df['jet_Wtrk3000'] < 0)]
    df_test_9 = train_df[(train_df['jet_Wtrk4000'] < 0)]

    ytrain_0 = df_train.jet_trk_nca
    ytrain_1 = df_train.jet_trk_ncasd
    ytrain_2 = df_train.jet_trk_rg
    ytrain_3 = df_train.jet_trk_zg
    ytrain_4 = df_train.jet_trk_c1beta02
    ytrain_5 = df_train.jet_Wtrk1000
    ytrain_6 = df_train.jet_Wtrk500
    ytrain_7 = df_train.jet_Wtrk2000
    ytrain_8 = df_train.jet_Wtrk3000
    ytrain_9 = df_train.jet_Wtrk4000

    Xtrain = df_train.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                            'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                            axis=1)

    ytest_0 = df_test_0.jet_trk_nca
    ytest_1 = df_test_1.jet_trk_ncasd
    ytest_2 = df_test_2.jet_trk_rg
    ytest_3 = df_test_3.jet_trk_zg
    ytest_4 = df_test_4.jet_trk_c1beta02
    ytest_5 = df_test_5.jet_Wtrk1000
    ytest_6 = df_test_6.jet_Wtrk500
    ytest_7 = df_test_7.jet_Wtrk2000
    ytest_8 = df_test_8.jet_Wtrk3000
    ytest_9 = df_test_9.jet_Wtrk4000

    Xtest_0= df_test_0.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_1= df_test_1.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_2= df_test_2.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_3= df_test_3.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_4= df_test_4.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_5= df_test_5.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_6= df_test_6.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_7= df_test_7.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_8= df_test_8.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)
    Xtest_9= df_test_9.drop(['jet_trk_nca','jet_trk_ncasd','jet_trk_rg','jet_trk_zg','jet_trk_c1beta02',
                             'jet_Wtrk1000','jet_Wtrk500','jet_Wtrk2000','jet_Wtrk3000','jet_Wtrk4000'],
                             axis=1)

    neigh0 = KNeighborsRegressor(n_neighbors=10)
    neigh1 = KNeighborsRegressor(n_neighbors=10)
    neigh2 = KNeighborsRegressor(n_neighbors=10)
    neigh3 = KNeighborsRegressor(n_neighbors=10)
    neigh4 = KNeighborsRegressor(n_neighbors=10)
    neigh5 = KNeighborsRegressor(n_neighbors=10)
    neigh6 = KNeighborsRegressor(n_neighbors=10)
    neigh7 = KNeighborsRegressor(n_neighbors=10)
    neigh8 = KNeighborsRegressor(n_neighbors=10)
    neigh9 = KNeighborsRegressor(n_neighbors=10)


    neigh0.fit(Xtrain, ytrain_0)
    neigh0.predict(Xtest_0,ytest_0)
    print ytest_0
    '''
    ytrain_1 = train_df.jet_trk_nca
    ytrain_2 = train_df.jet_trk_ncasd
    ytrain_3 = train_df.jet_trk_rg
    jet_trk_zg
    jet_trk_c1beta02
    jet_Wtrk1000
    jet_Wtrk500
    jet_Wtrk2000
    jet_Wtrk3000
    jet_Wtrk4000
    y1 = 
    '''


parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
FillNans(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh)