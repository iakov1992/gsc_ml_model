import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
import gc
import time
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser

warnings.filterwarnings("ignore")

def BuildCorrelationMap(jet_type,etaLow,etaHigh):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"
	
	trainName = folder_name + 'train_etaLow'+str(etaLow)+'_etaHigh'+str(etaHigh)+'.csv'
	train_df = pd.read_csv(trainName)
	train_df['jet_pt_response'] = train_df.jet_pt / train_df.jet_true_pt
	train_df.drop(['PtWeight', 'FlatWeight','jet_true_eta','jet_true_phi','jet_true_e',
				   'jet_respE', 'jet_respPt','jet_true_m', 'jet_JESE','jet_JESPt','jet_JESEta',
				   'jet_JESPhi','jet_JESMass','jet_true_pt'],axis=1,inplace=True)
	# create correlation map
	corr_map = train_df.corr()
	
	fileName = 'CorrelationMap_etaLow'+str(etaLow)+'_etaHigh'+str(etaHigh)+'_'+str(jet_type)+'.pdf'
	with PdfPages('CorrelationMap.pdf') as pdf:
		fig = plt.figure(figsize=(15,8))
		sns.heatmap(corr_map, annot=False, cmap=plt.cm.Reds)
		plt.show()
		pdf.savefig(fig)
		
	corr_df = pd.DataFrame()
	corr_df["feature"] = corr_map.jet_pt_response.index
	corr_df['target'] = corr_map.jet_pt_response.values
	corr_df.sort_values(by=['target'],inplace=True)
	i = corr_df[corr_df.feature == "jet_pt_response"].index
	corr_df.drop(i,axis=0, inplace=True)
	
	fileName = 'TargetVariableCorrelation_etaLow'+str(etaLow)+'_etaHigh'+str(etaHigh)+'_'+str(jet_type)+'.pdf'
	with PdfPages(fileName) as pdf:
		fig = plt.figure(figsize=(15,10))
		plt.barh(corr_df.feature,corr_df.target.values)
		plt.xlabel("Pearson Correlation with the Jet $p_{T}$ Response")
		plt.show()
		pdf.savefig(fig)



parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
BuildCorrelationMap(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh)