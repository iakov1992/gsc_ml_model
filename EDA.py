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



def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)
            
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Start memory usage is",start_mem_usg, ", Now memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df

# statistical parameters
def getWeightedMeanQuantiles(feature,weights):
    weighted_stats = DescrStatsW(feature, weights=weights, ddof=0)
    return weighted_stats.mean, weighted_stats.quantile([0.25,0.50,0.75],return_pandas=False)
def createFeatureDeviations(df,column):
    # feature deviations from the weighted mean and weighted quantiles
    print("create statistical features for", str(column))
    mean, quantiles = getWeightedMeanQuantiles(X_train[str(column)],PtWeight_train)
    q25 = quantiles[0]
    q50 = quantiles[1]
    q75 = quantiles[2]
    df[str(column)+'_deviation_mean'] = df[str(column)] - mean
    df[str(column)+'_abs_deviation_mean'] = np.abs(df[str(column)+'_deviation_mean'])
    df[str(column)+'_deviation_q25'] = df[str(column)] - q25
    df[str(column)+'_abs_deviation_q25'] = np.abs(df[str(column)+'_deviation_q25'])
    df[str(column)+'_deviation_q50'] = df[str(column)] - q50
    df[str(column)+'_abs_deviation_q50'] = np.abs(df[str(column)+'_deviation_q50'])
    df[str(column)+'_deviation_q75'] = df[str(column)] - q75
    df[str(column)+'_abs_deviation_q75'] = np.abs(df[str(column)+'_deviation_q75'])


def ExplaratoryDataAnalysis(jet_type,etaLow,etaHigh):
    folder_name = "jets"
    dataset_name = "totalDataFrame_etaLow_" + str(etaLow) + "_etaHigh_" + str(etaHigh)+".csv"
    if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
    dataFrameName = str(folder_name) + str(dataset_name)
    print "Process dataframe", dataFrameName
    dataFrame = pd.read_csv(dataFrameName)
    dataFrame.reset_index(inplace=True, drop=True)
    print("Split into train, test and validation datasets")
    train_df = dataFrame[:int(0.8*len(dataFrame))]
    test_df = dataFrame[int(0.8*len(dataFrame)):int(0.9*len(dataFrame))]
    validation_df = dataFrame[int(0.9*len(dataFrame)):]
    
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    validation_df.reset_index(inplace=True, drop=True)
    print "all",len(dataFrame)
    print "train",len(train_df)
    print "test",len(test_df)
    print "valid",len(validation_df)
    
    PtWeight_train = train_df.PtWeight
    SumWeights_train = train_df.SumWeights
    FlatWeight_train = train_df.FlatWeight
    jet_respE_train = train_df.jet_respE
    jet_respPt_train = train_df.jet_respPt
    
    PtWeight_test = test_df.PtWeight
    SumWeights_test = test_df.SumWeights
    FlatWeight_test = test_df.FlatWeight
    jet_respE_test = test_df.jet_respE
    jet_respPt_test = test_df.jet_respPt
    
    PtWeight_valid = validation_df.PtWeight
    SumWeights_valid = validation_df.SumWeights
    FlatWeight_valid = validation_df.FlatWeight
    jet_respE_valid = validation_df.jet_respE
    jet_respPt_valid = validation_df.jet_respPt
    
    print("Drop irrelevant columns")
    train_df.drop(['njet','jet_ChargedFraction','jet_trk_isd','PtWeight','SumWeights','FlatWeight','averageInteractionsPerCrossing','jet_Jvt','jet_ghostFrac','jet_n90Constituents','jet_PartonTruthLabelID',"jet_VoronoiArea"],axis=1,inplace=True)
    test_df.drop(['njet','jet_ChargedFraction','jet_trk_isd','PtWeight','SumWeights','FlatWeight','averageInteractionsPerCrossing','jet_Jvt','jet_ghostFrac','jet_n90Constituents','jet_PartonTruthLabelID',"jet_VoronoiArea"],axis=1,inplace=True)
    validation_df.drop(['njet','jet_ChargedFraction','jet_trk_isd', 'PtWeight','SumWeights','FlatWeight','averageInteractionsPerCrossing','jet_Jvt','jet_ghostFrac','jet_n90Constituents','jet_PartonTruthLabelID',"jet_VoronoiArea"],axis=1,inplace=True)
    
    train_df['abs_eta'] = np.abs(train_df['jet_eta'])
    test_df['abs_eta'] = np.abs(train_df['jet_eta'])
    validation_df['abs_eta'] = np.abs(train_df['jet_eta'])

    train_df['abs_DetEta'] = np.abs(train_df['jet_DetEta'])
    test_df['abs_DetEta'] = np.abs(train_df['jet_DetEta'])
    validation_df['abs_DetEta'] = np.abs(train_df['jet_DetEta'])

    train_df['abs_phi'] = np.abs(train_df['jet_phi'])
    test_df['abs_phi'] = np.abs(train_df['jet_phi'])
    validation_df['abs_phi'] = np.abs(train_df['jet_phi'])
    
    train_df['abs_ConstitEta'] = np.abs(train_df['jet_ConstitEta'])
    test_df['abs_ConstitEta'] = np.abs(train_df['jet_ConstitEta'])
    validation_df['abs_ConstitEta'] = np.abs(train_df['jet_ConstitEta'])

    train_df['abs_ConstitPhi'] = np.abs(train_df['jet_ConstitPhi'])
    test_df['abs_ConstitPhi'] = np.abs(train_df['jet_ConstitPhi'])
    validation_df['abs_ConstitPhi'] = np.abs(train_df['jet_ConstitPhi'])
    
    print("Make Plots")
    plotFolderName = ""
    if jet_type == "LCTopo":
        plotFolderName = "plots/LCTopo/"
    else:
        plotFolderName = "plots/UFO/"
    with PdfPages(plotFolderName + "UnweightedJetPtDist_"+"etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+".pdf") as pdf:
        fig = plt.figure(figsize=(20,10))
        plt.hist(train_df.jet_true_pt,bins=4000, histtype='step', label="Unweighted True")
        plt.hist(train_df.jet_pt,bins=4000, histtype='step', label="Unweghted Reco")
        plt.legend()
        plt.yscale('log')
        plt.ylabel('Unweighted Counts')
        plt.xlabel("Jet $p_{T}$")
        plt.show()
        pdf.savefig(fig)
        
    with PdfPages(plotFolderName + 'WeightedJetPtDist'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig = plt.figure(figsize=(20,10))
        plt.hist(train_df.jet_true_pt,bins=4000, histtype='step', weights=PtWeight_train, label="Weighted True")
        plt.hist(train_df.jet_pt,bins=4000, histtype='step', weights=PtWeight_train, label="Weighted Reco")
        plt.legend()
        plt.yscale('log')
        plt.ylabel('$p_{T}$ Weighted Counts')
        plt.xlabel("Jet $p_{T}$")
        plt.show()
        pdf.savefig(fig)
    
    with PdfPages(plotFolderName + 'FlatJetPtDist'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig = plt.figure(figsize=(20,10))
        plt.hist(train_df.jet_true_pt,bins=4000, histtype='step', weights=FlatWeight_train, label="Weighted (Flat) True")
        plt.hist(train_df.jet_pt,bins=4000, histtype='step', weights=FlatWeight_train, label="Weighted (Flat) Reco")
        plt.legend()
        plt.yscale('log')
        plt.ylabel('Flat Weighted Counts')
        plt.xlabel("Jet $p_{T}$")
        plt.show()
        pdf.savefig(fig)
    
    features = ['NPV', 'actualInteractionsPerCrossing', 'rho', 'jet_E', 'jet_pt',
                'jet_phi', 'jet_eta', 'jet_DetEta', 'jet_ConstitE', 'jet_nMuSeg',
                'jet_trk_nca', 'jet_trk_ncasd', 'jet_trk_rg', 'jet_trk_zg',
                'jet_trk_c1beta02']
    
    with PdfPages(plotFolderName + 'FeatureDistributions'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig,ax = plt.subplots(5,3,figsize=(30,30))
        varindex = 0
        for axlist in ax:
            for ax_i in axlist:
                varname = features[varindex]
                h = ax_i.hist(train_df[varname],bins=50,histtype='step',edgecolor='r',label='train',density=True,weights=PtWeight_train)
                binning = h[1]
                ax_i.hist(test_df[varname],bins=binning,histtype='step',edgecolor='b',label='test',density=True,weights=PtWeight_test)
                ax_i.legend()
                #ax_i.set_title(varname)
                ax_i.set_ylabel("weighted counts")
                ax_i.set_xlabel(varname)
                ax_i.set_yscale('log')
                varindex += 1
            
        plt.show()
        pdf.savefig(fig)
            

    features = ['jet_Ntrk500', 'jet_Ntrk1000', 'jet_Ntrk2000', 'jet_Ntrk3000', 'jet_Ntrk4000',
                'jet_Wtrk500', 'jet_Wtrk1000', 'jet_Wtrk2000', 'jet_Wtrk3000', 'jet_Wtrk4000',
                'rhoEM', 'jet_GhostArea', 'jet_ActiveArea', 'jet_true_m', 'jet_ActiveArea4vec_pt', 
                'jet_ActiveArea4vec_eta', 'jet_ActiveArea4vec_phi', 'jet_ActiveArea4vec_m', 'jet_m', 
                'jet_ConstitPt', 'jet_ConstitEta','jet_ConstitPhi','jet_ConstitMass',
                'jet_JESE','jet_JESPt','jet_JESMass','jet_JESEta','jet_JESPhi']

    with PdfPages(plotFolderName + 'FeatureDistributions2'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
	    fig,ax = plt.subplots(7,4,figsize=(30,30))
	    varindex = 0
	    for axlist in ax:
		    for ax_i in axlist:
			    varname = features[varindex]
			    h = ax_i.hist(train_df[varname],bins=50,histtype='step',edgecolor='r',label='train',density=True,weights=PtWeight_train)
			    binning = h[1]
			    ax_i.hist(test_df[varname],bins=binning,histtype='step',edgecolor='b',label='test',density=True,weights=PtWeight_test)
			    ax_i.legend()
		        #ax_i.set_title(varname)
			    ax_i.set_ylabel("weighted counts")
			    ax_i.set_xlabel(varname)
			    ax_i.set_yscale('log')
			    varindex += 1
	    plt.show()
	    pdf.savefig(fig)


    with PdfPages(plotFolderName + 'B-layers'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig,ax = plt.subplots(2,2,figsize=(20,20))
        ax[0][0].hist(dataFrame.jet_EnergyPerSampling_0,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_0.min()))
        ax[0][0].set_yscale("log")
        ax[0][0].set_ylabel("Counts, a.u.")
        ax[0][0].set_xlabel("PreSamplerB")
        ax[0][0].legend(loc="upper right")
        
        ax[0][1].hist(dataFrame.jet_EnergyPerSampling_1,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_1.min()))
        ax[0][1].set_yscale("log")
        ax[0][1].set_ylabel("Counts, a.u.")
        ax[0][1].set_xlabel("EMB1")
        ax[0][1].legend(loc="upper right")

        ax[1][0].hist(dataFrame.jet_EnergyPerSampling_2,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_2.min()))
        ax[1][0].set_yscale("log")
        ax[1][0].set_ylabel("Counts, a.u.")
        ax[1][0].set_xlabel("EMB2")
        ax[1][0].legend(loc="upper right")
        
        ax[1][1].hist(dataFrame.jet_EnergyPerSampling_3,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_3.min()))
        ax[1][1].set_yscale("log")
        ax[1][1].set_ylabel("Counts, a.u.")
        ax[1][1].set_xlabel("EMB3")
        ax[1][1].legend(loc="upper right")
        
        pdf.savefig(fig)

    with PdfPages(plotFolderName + 'E-layers'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig,ax = plt.subplots(2,2,figsize=(20,20))
        
        ax[0][0].hist(dataFrame.jet_EnergyPerSampling_4,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_4.min()))
        ax[0][0].set_yscale("log")
        ax[0][0].set_ylabel("Counts, a.u.")
        ax[0][0].set_xlabel("PreSamplerE")
        ax[0][0].legend(loc="upper right")
        
        ax[0][1].hist(dataFrame.jet_EnergyPerSampling_5,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_5.min()))
        ax[0][1].set_yscale("log")
        ax[0][1].set_ylabel("Counts, a.u.")
        ax[0][1].set_xlabel("EME1")
        ax[0][1].legend(loc="upper right")

        ax[1][0].hist(dataFrame.jet_EnergyPerSampling_6,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_6.min()))
        ax[1][0].set_yscale("log")
        ax[1][0].set_ylabel("Counts, a.u.")
        ax[1][0].set_xlabel("EME2")
        ax[1][0].legend(loc="upper right")
        
        ax[1][1].hist(dataFrame.jet_EnergyPerSampling_7,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_7.min()))
        ax[1][1].set_yscale("log")
        ax[1][1].set_ylabel("Counts, a.u.")
        ax[1][1].set_xlabel("EME3")
        ax[1][1].legend(loc="upper right")
        
        pdf.savefig(fig)


    with PdfPages(plotFolderName + 'HEC-layers'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig,ax = plt.subplots(2,2,figsize=(20,20))
    
        ax[0][0].hist(dataFrame.jet_EnergyPerSampling_8,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_8.min()))
        ax[0][0].set_yscale("log")
        ax[0][0].set_ylabel("Counts, a.u.")
        ax[0][0].set_xlabel("HEC0")
        ax[0][0].legend(loc="upper right")
        
        ax[0][1].hist(dataFrame.jet_EnergyPerSampling_9,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_9.min()))
        ax[0][1].set_yscale("log")
        ax[0][1].set_ylabel("Counts, a.u.")
        ax[0][1].set_xlabel("HEC1")
        ax[0][1].legend(loc="upper right")

        ax[1][0].hist(dataFrame.jet_EnergyPerSampling_10,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_10.min()))
        ax[1][0].set_yscale("log")
        ax[1][0].set_ylabel("Counts, a.u.")
        ax[1][0].set_xlabel("HEC2")
        ax[1][0].legend(loc="upper right")
        
        ax[1][1].hist(dataFrame.jet_EnergyPerSampling_11,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_11.min()))
        ax[1][1].set_yscale("log")
        ax[1][1].set_ylabel("Counts, a.u.")
        ax[1][1].set_xlabel("HEC3")
        ax[1][1].legend(loc="upper right")
        
        pdf.savefig(fig)


    with PdfPages(plotFolderName + 'TileBarGapExt-layers'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig,ax = plt.subplots(3,3,figsize=(20,20))
    
        ax[0][0].hist(dataFrame.jet_EnergyPerSampling_12,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_12.min()))
        ax[0][0].set_yscale("log")
        ax[0][0].set_ylabel("Counts, a.u.")
        ax[0][0].set_xlabel("TileBar0")
        ax[0][0].legend(loc="upper right")
        
        ax[0][1].hist(dataFrame.jet_EnergyPerSampling_13,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_13.min()))
        ax[0][1].set_yscale("log")
        ax[0][1].set_ylabel("Counts, a.u.")
        ax[0][1].set_xlabel("TileBar1")
        ax[0][1].legend(loc="upper right")

        ax[0][2].hist(dataFrame.jet_EnergyPerSampling_14,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_14.min()))
        ax[0][2].set_yscale("log")
        ax[0][2].set_ylabel("Counts, a.u.")
        ax[0][2].set_xlabel("TileBar2")
        ax[0][2].legend(loc="upper right")
        
        ax[1][0].hist(dataFrame.jet_EnergyPerSampling_15,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_15.min()))
        ax[1][0].set_yscale("log")
        ax[1][0].set_ylabel("Counts, a.u.")
        ax[1][0].set_xlabel("TileGap1")
        ax[1][0].legend(loc="upper right")
        
        
            
        ax[1][1].hist(dataFrame.jet_EnergyPerSampling_16,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_16.min()))
        ax[1][1].set_yscale("log")
        ax[1][1].set_ylabel("Counts, a.u.")
        ax[1][1].set_xlabel("TileGap2")
        ax[1][1].legend(loc="upper right")
        
            
        ax[1][2].hist(dataFrame.jet_EnergyPerSampling_17,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_17.min()))
        ax[1][2].set_yscale("log")
        ax[1][2].set_ylabel("Counts, a.u.")
        ax[1][2].set_xlabel("TileGap3")
        ax[1][2].legend(loc="upper right")
        
            
        ax[2][0].hist(dataFrame.jet_EnergyPerSampling_18,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_18.min()))
        ax[2][0].set_yscale("log")
        ax[2][0].set_ylabel("Counts, a.u.")
        ax[2][0].set_xlabel("TileExt0")
        ax[2][0].legend(loc="upper right")
        
            
        ax[2][1].hist(dataFrame.jet_EnergyPerSampling_19,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_19.min()))
        ax[2][1].set_yscale("log")
        ax[2][1].set_ylabel("Counts, a.u.")
        ax[2][1].set_xlabel("TileExt1")
        ax[2][1].legend(loc="upper right")
        
            
        ax[2][2].hist(dataFrame.jet_EnergyPerSampling_20,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_20.min()))
        ax[2][2].set_yscale("log")
        ax[2][2].set_ylabel("Counts, a.u.")
        ax[2][2].set_xlabel("TileExt2")
        ax[2][2].legend(loc="upper right")
        
        pdf.savefig(fig)


    with PdfPages(plotFolderName + 'FCal-layers'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig,ax = plt.subplots(1,3,figsize=(30,10))
    
        ax[0].hist(dataFrame.jet_EnergyPerSampling_21,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_21.min()))
        ax[0].set_yscale("log")
        ax[0].set_ylabel("Counts, a.u.")
        ax[0].set_xlabel("FCal0")
        ax[0].legend(loc="upper right")
        
        ax[1].hist(dataFrame.jet_EnergyPerSampling_22,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_22.min()))
        ax[1].set_yscale("log")
        ax[1].set_ylabel("Counts, a.u.")
        ax[1].set_xlabel("FCal1")
        ax[1].legend(loc="upper right")

        ax[2].hist(dataFrame.jet_EnergyPerSampling_23,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_23.min()))
        ax[2].set_yscale("log")
        ax[2].set_ylabel("Counts, a.u.")
        ax[2].set_xlabel("FCal2")
        ax[2].legend(loc="upper right")
        
        pdf.savefig(fig)


    with PdfPages(plotFolderName + 'MiniFcal-layers'+"_etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+'.pdf') as pdf:
        fig,ax = plt.subplots(2,2,figsize=(20,20))
    
        ax[0][0].hist(dataFrame.jet_EnergyPerSampling_24,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_24.min()))
        ax[0][0].set_yscale("log")
        ax[0][0].set_ylabel("Counts, a.u.")
        ax[0][0].set_xlabel("MiniFcal0")
        ax[0][0].legend(loc="upper right")
        
        ax[0][1].hist(dataFrame.jet_EnergyPerSampling_25,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_25.min()))
        ax[0][1].set_yscale("log")
        ax[0][1].set_ylabel("Counts, a.u.")
        ax[0][1].set_xlabel("MiniFcal1")
        ax[0][1].legend(loc="upper right")

        ax[1][0].hist(dataFrame.jet_EnergyPerSampling_26,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_26.min()))
        ax[1][0].set_yscale("log")
        ax[1][0].set_ylabel("Counts, a.u.")
        ax[1][0].set_xlabel("MiniFcal2")
        ax[1][0].legend(loc="upper right")
        
        ax[1][1].hist(dataFrame.jet_EnergyPerSampling_27,bins=100,histtype='step',weights=dataFrame.PtWeight,label="min value: "+str(dataFrame.jet_EnergyPerSampling_27.min()))
        ax[1][1].set_yscale("log")
        ax[1][1].set_ylabel("Counts, a.u.")
        ax[1][1].set_xlabel("MiniFcal3")
        ax[1][1].legend(loc="upper right")
        
        pdf.savefig(fig)


    train_df['PtWeight'] = PtWeight_train
    test_df['PtWeight'] = PtWeight_test
    validation_df['PtWeight'] = PtWeight_valid

    train_df['FlatWeight'] = FlatWeight_train
    test_df['FlatWeight'] = FlatWeight_test
    validation_df['FlatWeight'] = FlatWeight_valid

    #print "Reduce memory usage one more time, to make surre that everything is optimal"
    #X_train = reduce_mem_usage(X_train)
    #X_test = reduce_mem_usage(X_test)
    #X_val = reduce_mem_usage(X_val)
 
    print "Create csv files"
    train_name = str(folder_name) + "train_etaLow"+str(etaLow)+"_etaHigh"+str(etaHigh)+".csv"
    test_name = str(folder_name) + "test_etaLow"+str(etaLow)+"_etaHigh"+str(etaHigh)+".csv"
    valid_name = str(folder_name) + "valid_etaLow"+str(etaLow)+"_etaHigh"+str(etaHigh)+".csv"
    train_df.to_csv(train_name,index=False)
    test_df.to_csv(test_name,index=False)
    validation_df.to_csv(valid_name,index=False)


parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
ExplaratoryDataAnalysis(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh)