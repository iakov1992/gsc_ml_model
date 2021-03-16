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
from statsmodels.stats.weightstats import DescrStatsW
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
warnings.filterwarnings("ignore")


def PlotDistributions(jet_type,etaLow,etaHigh,scaledDataFrame):
    folder_name = "jets"
    if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"

    if scaledDataFrame == "yes":
        folder_name = folder_name + "scaled_"
    train_df = pd.read_csv(folder_name + "train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
    test_df = pd.read_csv(folder_name + "test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
    valid_df = pd.read_csv(folder_name + "valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")


    plotFolderName = "noName"
    if jet_type == "LCTopo":
        plotFolderName = "/srv01/cgrp/iakova/PhD/QualificationTask/2020_10_08/ExploratoryDataAnalysis/plots/noNan/LCTopo/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
        if scaledDataFrame == "yes":
            plotFolderName = "/srv01/cgrp/iakova/PhD/QualificationTask/2020_10_08/ExploratoryDataAnalysis/plots/noNan/LCTopo/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"+"scaled/"
    else:
        plotFolderName = "/srv01/cgrp/iakova/PhD/QualificationTask/2020_10_08/ExploratoryDataAnalysis/plots/noNan/UFO/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
        if scaledDataFrame == "yes":
            plotFolderName = "/srv01/cgrp/iakova/PhD/QualificationTask/2020_10_08/ExploratoryDataAnalysis/plots/noNan/UFO/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"+"scaled/"

    if os.path.exists(plotFolderName):
        print "Folder",plotFolderName,"already exists"
    else:
        os.mkdir(plotFolderName)

    for col in train_df.columns:
        if col == "PtWeight":
            continue
        if col == "FlatWeight":
            continue
        with PdfPages(plotFolderName + str(col)+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
            print "Process column", col
            fig = plt.figure(figsize=(20,10))
            h = plt.hist(train_df[str(col)],bins=100, histtype='step', weights=train_df.PtWeight, density=True, label="Train")
            binning = h[1]
            plt.hist(test_df[str(col)],bins=binning, histtype='step', weights=test_df.PtWeight, density=True, label="Test")
            plt.hist(valid_df[str(col)],bins=binning, histtype='step', weights=valid_df.PtWeight, density=True, label="Validation")
            plt.yscale('log')
            plt.ylabel('Weighted Counts')
            plt.xlabel(str(col))
            plt.legend()
            plt.show()
            pdf.savefig(fig)
        
parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--scaledDataFrame', default = "no", type = "string", help = "plots from scaled dataframe")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
PlotDistributions(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh,scaledDataFrame=opt.scaledDataFrame)