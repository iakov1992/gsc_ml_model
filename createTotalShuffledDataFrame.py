import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from optparse import OptionParser



def create_dataframe(etaLow,etaHigh,jet_type):
    folder_name = "jets"
    if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"

    print "jet_type =", jet_type,"etaLow =",etaLow,"etaHigh =",etaHigh
    samples = [364701, 364702, 364703, 364704, 364705, 364706, 364707, 364708, 364709, 364710, 364711, 364712]
    features = ["PtWeight","SumWeights","FlatWeight","NPV",
                "actualInteractionsPerCrossing", "averageInteractionsPerCrossing",
                "rho", "njet", "jet_E", "jet_pt", "jet_phi", "jet_eta", "jet_DetEta",
                "jet_Jvt", "jet_true_pt", "jet_true_eta", "jet_true_phi", "jet_true_e",
                "jet_respE", "jet_respPt", "jet_PartonTruthLabelID", "jet_Wtrk1000",
                "jet_Ntrk1000", "jet_ConstitE", "jet_ChargedFraction", "jet_nMuSeg",
                "jet_trk_isd", "jet_trk_nca", "jet_trk_ncasd", "jet_trk_rg", "jet_trk_zg",
                "jet_trk_c1beta02","jet_EnergyPerSampling_0","jet_EnergyPerSampling_1",
                "jet_EnergyPerSampling_2","jet_EnergyPerSampling_3","jet_EnergyPerSampling_4",
                "jet_EnergyPerSampling_5","jet_EnergyPerSampling_6","jet_EnergyPerSampling_7",
                "jet_EnergyPerSampling_8","jet_EnergyPerSampling_9","jet_EnergyPerSampling_10",
                "jet_EnergyPerSampling_11","jet_EnergyPerSampling_12","jet_EnergyPerSampling_13",
                "jet_EnergyPerSampling_14","jet_EnergyPerSampling_15","jet_EnergyPerSampling_16",
                "jet_EnergyPerSampling_17","jet_EnergyPerSampling_18","jet_EnergyPerSampling_19",
                "jet_EnergyPerSampling_20","jet_EnergyPerSampling_21","jet_EnergyPerSampling_22",
                "jet_EnergyPerSampling_23","jet_EnergyPerSampling_24","jet_EnergyPerSampling_25",
                "jet_EnergyPerSampling_26","jet_EnergyPerSampling_27","rhoEM", "jet_GhostArea",
                "jet_ActiveArea","jet_VoronoiArea","jet_ActiveArea4vec_pt","jet_ActiveArea4vec_eta",
                "jet_ActiveArea4vec_phi","jet_ActiveArea4vec_m","jet_n90Constituents","jet_m",
                "jet_true_m","jet_Ntrk500","jet_Wtrk500","jet_Ntrk2000","jet_Ntrk3000",
                "jet_Ntrk4000","jet_Wtrk2000","jet_Wtrk3000","jet_Wtrk4000","jet_ConstitPt",
                "jet_ConstitEta","jet_ConstitPhi","jet_ConstitMass","jet_JESE","jet_JESPt",
                "jet_JESEta","jet_JESPhi","jet_JESMass","jet_iso_dR","jet_ghostFrac"]
    
    df_array = []
    for sample in samples:
        print "sample = ",  sample
        df = pd.DataFrame()
        for feature in features:
            print "Add feature:", feature
            name = folder_name + "mc_sample_" + str(sample) + "_" + feature + ".hdf5"
            f = h5py.File(name,'r+')
            for key in f.keys():
                df[str(key)] = f[str(key)].value
        df_array.append(df)
    
    dataFrame = df_array[0]
    
    for index in range(1,len(df_array)):
        print "index =", index
        dataFrame = pd.concat([dataFrame, df_array[index]], axis=0, ignore_index=True)
        print "DataFrame length =", len(dataFrame)
        
    dataFrame = shuffle(dataFrame)
    
    dataset_name = "totalDataFrame_etaLow_" + str(etaLow) + "_etaHigh_" + str(etaHigh)+".csv"
    if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
    
    dataFrame.to_csv(str(folder_name) + str(dataset_name),index=False)
    print "Done!!!"


parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
create_dataframe(etaLow = opt.etaLow,etaHigh=opt.etaHigh,jet_type=opt.jet_type)