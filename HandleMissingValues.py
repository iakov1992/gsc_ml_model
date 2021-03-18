import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.utils import shuffle
import os
import warnings
import gc
import time
from sklearn import metrics
from optparse import OptionParser

warnings.filterwarnings("ignore")


def handleMissingValues(jet_type,etaLow,etaHigh):
    folder_name = "jets"
    if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
    train_df = pd.read_csv(folder_name + "train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + ".csv")
    test_df = pd.read_csv(folder_name + "test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + ".csv")
    valid_df = pd.read_csv(folder_name + "valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + ".csv")

    
    for col in train_df.columns:
        if len(train_df[str(col)].unique()) < 10:
            print ("drop column", col)
            train_df.drop(str(col),axis = 'columns',inplace=True)
            test_df.drop(str(col),axis = 'columns',inplace=True)
            valid_df.drop(str(col),axis = 'columns',inplace=True)
    
    print ("Selected Columns:")
    for col in train_df.columns:
        print (col)
    
    print "##########################################################################"
    print "fraction of missing values in jet_trk_nca in the training sample:", float(len(train_df[train_df["jet_trk_nca"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_nca in the test sample:", float(len(test_df[test_df["jet_trk_nca"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_nca in the validation sample:", float(len(valid_df[valid_df["jet_trk_nca"]<0.]))/float(len(valid_df))
    
    print "##########################################################################"
    print "fraction of missing values in jet_trk_ncasd in the training sample:", float(len(train_df[train_df["jet_trk_ncasd"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_ncasd in the test sample:", float(len(test_df[test_df["jet_trk_ncasd"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_ncasd in the validation sample:", float(len(valid_df[valid_df["jet_trk_ncasd"]<0.]))/float(len(valid_df))

    print "##########################################################################"
    print "fraction of missing values in jet_trk_rg in the training sample:", float(len(train_df[train_df["jet_trk_rg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_rg in the test sample:", float(len(test_df[test_df["jet_trk_rg"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_rg in the validation sample:", float(len(valid_df[valid_df["jet_trk_rg"]<0.]))/float(len(valid_df))

    print "##########################################################################"
    print "fraction of missing values in jet_trk_zg in the training sample:", float(len(train_df[train_df["jet_trk_zg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_zg in the test sample:", float(len(test_df[test_df["jet_trk_zg"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_zg in the validation sample:", float(len(valid_df[valid_df["jet_trk_zg"]<0.]))/float(len(valid_df))

    print "##########################################################################"
    print "fraction of missing values in jet_trk_c1beta02 in the training sample:", float(len(train_df[train_df["jet_trk_c1beta02"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_c1beta02 in the test sample:", float(len(test_df[test_df["jet_trk_c1beta02"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_c1beta02 in the validation sample:", float(len(valid_df[valid_df["jet_trk_c1beta02"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_nMuSeg in the training sample:", float(len(train_df[train_df["jet_nMuSeg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_nMuSeg in the test sample:", float(len(test_df[test_df["jet_nMuSeg"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_nMuSeg in the validation sample:", float(len(valid_df[valid_df["jet_nMuSeg"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in NPV in the training sample:", float(len(train_df[train_df["NPV"]<0.]))/float(len(train_df))
    print "fraction of missing values in NPV in the test sample:", float(len(test_df[test_df["NPV"]<0.]))/float(len(test_df))
    print "fraction of missing values in NPV in the validation sample:", float(len(valid_df[valid_df["NPV"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk1000 in the training sample:", float(len(train_df[train_df["jet_Wtrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk1000 in the test sample:", float(len(test_df[test_df["jet_Wtrk1000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk1000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk1000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk1000 in the training sample:", float(len(train_df[train_df["jet_Ntrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk1000 in the test sample:", float(len(test_df[test_df["jet_Ntrk1000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk1000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk1000"]<0.]))/float(len(valid_df))
    print "##########################################################################"



    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk500 in the training sample:", float(len(train_df[train_df["jet_Wtrk500"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk500 in the test sample:", float(len(test_df[test_df["jet_Wtrk500"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk500 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk500"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk500 in the training sample:", float(len(train_df[train_df["jet_Ntrk500"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk500 in the test sample:", float(len(test_df[test_df["jet_Ntrk500"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk500 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk500"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk2000 in the training sample:", float(len(train_df[train_df["jet_Wtrk2000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk2000 in the test sample:", float(len(test_df[test_df["jet_Wtrk2000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk2000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk2000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk2000 in the training sample:", float(len(train_df[train_df["jet_Ntrk2000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk2000 in the test sample:", float(len(test_df[test_df["jet_Ntrk2000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk2000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk2000"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk3000 in the training sample:", float(len(train_df[train_df["jet_Wtrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk3000 in the test sample:", float(len(test_df[test_df["jet_Wtrk1000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk3000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk1000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk3000 in the training sample:", float(len(train_df[train_df["jet_Ntrk3000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk3000 in the test sample:", float(len(test_df[test_df["jet_Ntrk3000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk3000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk3000"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk4000 in the training sample:", float(len(train_df[train_df["jet_Wtrk4000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk4000 in the test sample:", float(len(test_df[test_df["jet_Wtrk4000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk4000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk4000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk4000 in the training sample:", float(len(train_df[train_df["jet_Ntrk4000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk4000 in the test sample:", float(len(test_df[test_df["jet_Ntrk4000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk4000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk4000"]<0.]))/float(len(valid_df))
    print "##########################################################################"
    
    
    train_df.loc[train_df.jet_trk_rg < 0.,"jet_trk_rg"] = np.nan
    test_df.loc[test_df.jet_trk_rg < 0.,"jet_trk_rg"] = np.nan
    valid_df.loc[valid_df.jet_trk_rg < 0.,"jet_trk_rg"] = np.nan

    train_df.loc[train_df.jet_trk_zg < 0.,"jet_trk_zg"] = np.nan
    test_df.loc[test_df.jet_trk_zg < 0.,"jet_trk_zg"] = np.nan
    valid_df.loc[valid_df.jet_trk_zg < 0.,"jet_trk_zg"] = np.nan

    train_df.loc[train_df.jet_trk_nca < 0.,"jet_trk_nca"] = np.nan
    test_df.loc[test_df.jet_trk_nca < 0.,"jet_trk_nca"] = np.nan
    valid_df.loc[valid_df.jet_trk_nca < 0.,"jet_trk_nca"] = np.nan

    train_df.loc[train_df.jet_trk_ncasd < 0.,"jet_trk_ncasd"] = np.nan
    test_df.loc[test_df.jet_trk_ncasd < 0.,"jet_trk_ncasd"] = np.nan
    valid_df.loc[valid_df.jet_trk_ncasd < 0.,"jet_trk_ncasd"] = np.nan

    train_df.loc[train_df.jet_trk_c1beta02 < 0.,"jet_trk_c1beta02"] = np.nan
    test_df.loc[test_df.jet_trk_c1beta02 < 0.,"jet_trk_c1beta02"] = np.nan
    valid_df.loc[valid_df.jet_trk_c1beta02 < 0.,"jet_trk_c1beta02"] = np.nan


    train_df.loc[train_df.NPV < 0.,"NPV"] = np.nan
    test_df.loc[test_df.NPV < 0.,"NPV"] = np.nan
    valid_df.loc[valid_df.NPV < 0.,"NPV"] = np.nan
    

    train_df.loc[train_df.jet_Wtrk1000 < 0.,"jet_Wtrk1000"] = np.nan
    test_df.loc[test_df.jet_Wtrk1000 < 0.,"jet_Wtrk1000"] = np.nan
    valid_df.loc[valid_df.jet_Wtrk1000 < 0.,"jet_Wtrk1000"] = np.nan

    train_df.loc[train_df.jet_Ntrk1000 < 0.,"jet_Ntrk1000"] = np.nan
    test_df.loc[test_df.jet_Ntrk1000 < 0.,"jet_Ntrk1000"] = np.nan
    valid_df.loc[valid_df.jet_Ntrk1000 < 0.,"jet_Ntrk1000"] = np.nan

    train_df.loc[train_df.jet_Wtrk500 < 0.,"jet_Wtrk500"] = np.nan
    test_df.loc[test_df.jet_Wtrk500 < 0.,"jet_Wtrk500"] = np.nan
    valid_df.loc[valid_df.jet_Wtrk500 < 0.,"jet_Wtrk500"] = np.nan

    train_df.loc[train_df.jet_Ntrk500 < 0.,"jet_Ntrk500"] = np.nan
    test_df.loc[test_df.jet_Ntrk500 < 0.,"jet_Ntrk500"] = np.nan
    valid_df.loc[valid_df.jet_Ntrk500 < 0.,"jet_Ntrk500"] = np.nan

    train_df.loc[train_df.jet_Wtrk2000 < 0.,"jet_Wtrk2000"] = np.nan
    test_df.loc[test_df.jet_Wtrk2000 < 0.,"jet_Wtrk2000"] = np.nan
    valid_df.loc[valid_df.jet_Wtrk2000 < 0.,"jet_Wtrk2000"] = np.nan

    train_df.loc[train_df.jet_Ntrk2000 < 0.,"jet_Ntrk2000"] = np.nan
    test_df.loc[test_df.jet_Ntrk2000 < 0.,"jet_Ntrk2000"] = np.nan
    valid_df.loc[valid_df.jet_Ntrk2000 < 0.,"jet_Ntrk2000"] = np.nan

    train_df.loc[train_df.jet_Wtrk3000 < 0.,"jet_Wtrk3000"] = np.nan
    test_df.loc[test_df.jet_Wtrk3000 < 0.,"jet_Wtrk3000"] = np.nan
    valid_df.loc[valid_df.jet_Wtrk3000 < 0.,"jet_Wtrk3000"] = np.nan

    train_df.loc[train_df.jet_Ntrk3000 < 0.,"jet_Ntrk3000"] = np.nan
    test_df.loc[test_df.jet_Ntrk3000 < 0.,"jet_Ntrk3000"] = np.nan
    valid_df.loc[valid_df.jet_Ntrk3000 < 0.,"jet_Ntrk3000"] = np.nan

    train_df.loc[train_df.jet_Wtrk4000 < 0.,"jet_Wtrk4000"] = np.nan
    test_df.loc[test_df.jet_Wtrk4000 < 0.,"jet_Wtrk4000"] = np.nan
    valid_df.loc[valid_df.jet_Wtrk4000 < 0.,"jet_Wtrk4000"] = np.nan

    train_df.loc[train_df.jet_Ntrk4000 < 0.,"jet_Ntrk4000"] = np.nan
    test_df.loc[test_df.jet_Ntrk4000 < 0.,"jet_Ntrk4000"] = np.nan
    valid_df.loc[valid_df.jet_Ntrk4000 < 0.,"jet_Ntrk4000"] = np.nan
   

    print ("Nans train: ",train_df.isna().sum())
    print ("Nans test: ",test_df.isna().sum())
    print ("Nans valid: ",valid_df.isna().sum())

    train_df.dropna(axis=0,inplace=True)
    test_df.dropna(axis=0,inplace=True)
    valid_df.dropna(axis=0,inplace=True)

    print "After dropping Nans"
    print ("Nans train: ",train_df.isna().sum())
    print ("Nans test: ",test_df.isna().sum())
    print ("Nans valid: ",valid_df.isna().sum())


    print "##########################################################################"
    print "##################### After dropping Nans  ###############################"
    print "##########################################################################"
    print "fraction of missing values in jet_trk_nca in the training sample:", float(len(train_df[train_df["jet_trk_nca"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_nca in the test sample:", float(len(test_df[test_df["jet_trk_nca"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_nca in the validation sample:", float(len(valid_df[valid_df["jet_trk_nca"]<0.]))/float(len(valid_df))
    
    print "##########################################################################"
    print "fraction of missing values in jet_trk_ncasd in the training sample:", float(len(train_df[train_df["jet_trk_ncasd"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_ncasd in the test sample:", float(len(test_df[test_df["jet_trk_ncasd"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_ncasd in the validation sample:", float(len(valid_df[valid_df["jet_trk_ncasd"]<0.]))/float(len(valid_df))

    print "##########################################################################"
    print "fraction of missing values in jet_trk_rg in the training sample:", float(len(train_df[train_df["jet_trk_rg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_rg in the test sample:", float(len(test_df[test_df["jet_trk_rg"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_rg in the validation sample:", float(len(valid_df[valid_df["jet_trk_rg"]<0.]))/float(len(valid_df))

    print "##########################################################################"
    print "fraction of missing values in jet_trk_zg in the training sample:", float(len(train_df[train_df["jet_trk_zg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_zg in the test sample:", float(len(test_df[test_df["jet_trk_zg"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_zg in the validation sample:", float(len(valid_df[valid_df["jet_trk_zg"]<0.]))/float(len(valid_df))

    print "##########################################################################"
    print "fraction of missing values in jet_trk_c1beta02 in the training sample:", float(len(train_df[train_df["jet_trk_c1beta02"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_trk_c1beta02 in the test sample:", float(len(test_df[test_df["jet_trk_c1beta02"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_trk_c1beta02 in the validation sample:", float(len(valid_df[valid_df["jet_trk_c1beta02"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_nMuSeg in the training sample:", float(len(train_df[train_df["jet_nMuSeg"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_nMuSeg in the test sample:", float(len(test_df[test_df["jet_nMuSeg"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_nMuSeg in the validation sample:", float(len(valid_df[valid_df["jet_nMuSeg"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in NPV in the training sample:", float(len(train_df[train_df["NPV"]<0.]))/float(len(train_df))
    print "fraction of missing values in NPV in the test sample:", float(len(test_df[test_df["NPV"]<0.]))/float(len(test_df))
    print "fraction of missing values in NPV in the validation sample:", float(len(valid_df[valid_df["NPV"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk1000 in the training sample:", float(len(train_df[train_df["jet_Wtrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk1000 in the test sample:", float(len(test_df[test_df["jet_Wtrk1000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk1000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk1000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk1000 in the training sample:", float(len(train_df[train_df["jet_Ntrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk1000 in the test sample:", float(len(test_df[test_df["jet_Ntrk1000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk1000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk1000"]<0.]))/float(len(valid_df))
    print "##########################################################################"



    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk500 in the training sample:", float(len(train_df[train_df["jet_Wtrk500"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk500 in the test sample:", float(len(test_df[test_df["jet_Wtrk500"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk500 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk500"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk500 in the training sample:", float(len(train_df[train_df["jet_Ntrk500"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk500 in the test sample:", float(len(test_df[test_df["jet_Ntrk500"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk500 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk500"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk2000 in the training sample:", float(len(train_df[train_df["jet_Wtrk2000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk2000 in the test sample:", float(len(test_df[test_df["jet_Wtrk2000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk2000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk2000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk2000 in the training sample:", float(len(train_df[train_df["jet_Ntrk2000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk2000 in the test sample:", float(len(test_df[test_df["jet_Ntrk2000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk2000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk2000"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk3000 in the training sample:", float(len(train_df[train_df["jet_Wtrk1000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk3000 in the test sample:", float(len(test_df[test_df["jet_Wtrk1000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk3000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk1000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk3000 in the training sample:", float(len(train_df[train_df["jet_Ntrk3000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk3000 in the test sample:", float(len(test_df[test_df["jet_Ntrk3000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk3000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk3000"]<0.]))/float(len(valid_df))
    print "##########################################################################"


    print "##########################################################################"
    print "fraction of missing values in jet_Wtrk4000 in the training sample:", float(len(train_df[train_df["jet_Wtrk4000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Wtrk4000 in the test sample:", float(len(test_df[test_df["jet_Wtrk4000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Wtrk4000 in the validation sample:", float(len(valid_df[valid_df["jet_Wtrk4000"]<0.]))/float(len(valid_df))
    print "##########################################################################"

    print "##########################################################################"
    print "fraction of missing values in jet_Ntrk4000 in the training sample:", float(len(train_df[train_df["jet_Ntrk4000"]<0.]))/float(len(train_df))
    print "fraction of missing values in jet_Ntrk4000 in the test sample:", float(len(test_df[test_df["jet_Ntrk4000"]<0.]))/float(len(test_df))
    print "fraction of missing values in jet_Ntrk4000 in the validation sample:", float(len(valid_df[valid_df["jet_Ntrk4000"]<0.]))/float(len(valid_df))
    print "##########################################################################"
    

    print ("# of rows in training sample:", len(train_df))
    print ("# of rows in test sample:", len(test_df))
    print ("# of rows in validation sample:", len(valid_df))

    print ("Shape of the training sample:", train_df.shape)
    print ("Shape of the test sample:", test_df.shape)
    print ("Shape of the validation sample:", valid_df.shape)
    
    print ("Create csv files")
    train_df.to_csv(folder_name + "train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv",index=False)
    test_df.to_csv(folder_name + "test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv",index=False)
    valid_df.to_csv(folder_name + "valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv",index=False)
    
    

parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
handleMissingValues(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh)