from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import ROOT as r 
import os
import warnings
import gc
import settings as settings
from array import array
import commonCode as cc
import JES_BalanceFitter as jbf
import AtlasStyle as astyle
from optparse import OptionParser

warnings.filterwarnings("ignore")


def mape(y_true,y_pred,sample_weight):
	err = 0.
	total_weight = 0.
	for i in range(len(y_true)):
		sample_error = np.abs(sample_weight[i]*(y_true[i] - y_pred[i])/y_true[i])
		err += sample_error
		total_weight += sample_weight[i]
	wmape = 100 * err / total_weight
	print "wmape =", wmape
	return wmape


def msape(y_true, y_pred):
	return K.mean(np.divide((y_pred-y_true), y_true)*np.divide((y_pred-y_true), y_true), axis=-1)

def drawHists(hists, fileName, rangeMin, rangeMax, drawOptions, legend, colors, islog):
	c1 = ROOT.TCanvas("Canvas%s"%fileName,"",800,600)
	if islog:
		c1.SetLogy()
	drawOptions = ""
	index=1
	if len(hists) == 0:
		return
	for hist in hists:
		hist.GetYaxis().SetRangeUser(rangeMin, rangeMax)
    	hist.GetXaxis().SetRangeUser(20., 60.)
    	testlen = len(colors)
    	hist.SetLineColor(ROOT.TColor.GetColor(colors[(index-1)%testlen]))
    	hist.SetMarkerSize(0)
    	hist.Draw(drawOptions)
    	drawOptions = drawOptions + "SAME"
    	index+=1
	legend.Draw()
	#cc.draw_atlas_details(labels=labels)
	c1.Print(fileName+".pdf")


def passesCut(x, binlow, binhigh):
	if x < binhigh and x >= binlow:
		return 1
	return 0

def getPtBin(jetPt):
	for cbin in range(len(settings.ptBins)):
		if jetPt < settings.ptBins[cbin]:
			return cbin -1
	return -1

def getBin(value, bins):
	for cbin in range(len(bins)):
		if value < bins[cbin]:
			return cbin -1
	return -1


def getResponseFit(hist):
	fitter = jbf.JES_BalanceFitter(1.5)
	if hist.Integral() == 0:
		return 0, 0, 0, 0, 0
	myFit =  fitter.Fit(hist)
	mean  = myFit.GetParameter(1)
	sigma = myFit.GetParameter(2)
	meanError  = myFit.GetParError(1)
	sigmaError = myFit.GetParError(2)
	myFit.SetLineColor(ROOT.kRed)

	return myFit, mean, sigma, meanError, sigmaError


astyle.SetAtlasStyle()


def MakePlots(jet_type,etaLow,etaHigh,scale_target,variable_type):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/"

	fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_20201222.csv'
	fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_20201222.csv'
	fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_20201222.csv'
	if scale_target == "no":
		fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_targetIsNotScaled_20201222.csv'
		fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_targetIsNotScaled_20201222.csv'
		fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_targetIsNotScaled_20201222.csv'
	if variable_type == "calorimeter":
		fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_20201222.csv'
		fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_20201222.csv'
		fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_20201222.csv'
		if scale_target == "no":
			fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_targetIsNotScaled_20201222.csv'
			fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_targetIsNotScaled_20201222.csv'
			fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_calorimeter_targetIsNotScaled_20201222.csv'
	if variable_type == "nocalorimeter":
		fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_20201222.csv'
		fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_20201222.csv'
		fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_20201222.csv'
		if scale_target == "no":
			fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_targetIsNotScaled_20201222.csv'
			fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_targetIsNotScaled_20201222.csv'
			fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_nocalorimeter_targetIsNotScaled_20201222.csv'

	if variable_type == "all":
		fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_20201222.csv'
		fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_20201222.csv'
		fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_20201222.csv'
		if scale_target == "no":
			fileNameTrain = folder_name+'scaled_prediction_train_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_targetIsNotScaled_20201222.csv'
			fileNameTest = folder_name+'scaled_prediction_test_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_targetIsNotScaled_20201222.csv'
			fileNameValid = folder_name+'scaled_prediction_valid_df_etaLow_'+str(etaLow)+'_etaHigh_'+str(etaHigh)+'_all_targetIsNotScaled_20201222.csv'


	saveFolder = "/srv01/cgrp/iakova/PhD/QualificationTask/2021_01_01/NeuralNetworkModels/"+str(jet_type)+"/"+str(variable_type)+"/scale_"+str(scale_target)+"/etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh)
	if os.path.exists(saveFolder):
		os.rmdir(saveFolder)
	print "Create folder:", saveFolder
	os.makedirs(saveFolder)
	suffix = '20201224'
	print "Read dataframes"
	train_df = pd.read_csv(folder_name + "train_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
	test_df = pd.read_csv(folder_name + "test_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
	valid_df = pd.read_csv(folder_name + "valid_etaLow" + str(etaLow) + "_etaHigh" + str(etaHigh) + "_noNan.csv")
	scaled_df_pred_train = pd.read_csv(fileNameTrain)
	scaled_df_pred_test = pd.read_csv(fileNameTest)
	scaled_df_pred_valid = pd.read_csv(fileNameValid)

	print train_df.shape, scaled_df_pred_train.shape
	print test_df.shape, scaled_df_pred_test.shape
	print valid_df.shape, scaled_df_pred_valid.shape

	df_array = [scaled_df_pred_train,scaled_df_pred_test,scaled_df_pred_valid]
	#df_array2 = [train_df,test_df,valid_df]
	dataFrame = df_array[0]
	#dataFrame2 = df_array2[0]
	for index in range(1,len(df_array)):
		print "index =", index
		dataFrame = pd.concat([dataFrame, df_array[index]], axis=0, ignore_index=True)
		#dataFrame2 = pd.concat([dataFrame2, df_array2[index]], axis=0, ignore_index=True)


	train_df['jet_pt_response'] = train_df['jet_pt'] / train_df['jet_true_pt']

	max_jet_true_pt = train_df['jet_true_pt'].max()
	max_jet_pt = train_df['jet_pt'].max()
	max_jet_pt_response = train_df['jet_pt_response'].max()

	min_jet_true_pt = train_df['jet_true_pt'].min()
	min_jet_pt = train_df['jet_pt'].min()
	min_jet_pt_response = train_df['jet_pt_response'].min()

	min_jet_JES_pt = train_df['jet_JESPt'].min()
	max_jet_JES_pt = train_df['jet_JESPt'].max()

	print "dataFrame shape", dataFrame.shape
	print "max_jet_true_pt =",max_jet_true_pt,", max_jet_pt =",max_jet_pt
	print "min_jet_true_pt =",train_df['jet_true_pt'].min(),", min_jet_pt =",train_df['jet_pt'].min()
	'''
	jet_true_pt = dataFrame["jet_true_pt"] * (max_jet_true_pt - min_jet_true_pt) + min_jet_true_pt
	jet_pt = dataFrame["jet_pt"] * (max_jet_pt - min_jet_pt) + min_jet_pt
	jet_JESPt = dataFrame["jet_JESPt"]  * (max_jet_JES_pt - min_jet_JES_pt) + min_jet_JES_pt
	'''
	jet_true_pt = dataFrame["jet_true_pt"] * max_jet_true_pt
	jet_pt = dataFrame["jet_pt"] * max_jet_pt
	jet_JESPt = dataFrame["jet_JESPt"] * max_jet_JES_pt
	PtWeight = dataFrame["PtWeight"]
	FlatWeight = dataFrame["FlatWeight"]

	jet_pt_response_olm1 = dataFrame['y_olm1']
	jet_pt_response_olm2 = dataFrame['y_olm2']
	jet_pt_response_olm3 = dataFrame['y_olm3']
	jet_pt_response_olm = dataFrame['y_olm1'] / 3 + dataFrame['y_olm2'] / 3 + dataFrame['y_olm3'] / 3
	#jet_pt_response_olm = jet_pt_response_olm2

	jet_pt_olm1 = jet_pt / jet_pt_response_olm1
	jet_pt_olm2 = jet_pt / jet_pt_response_olm2
	jet_pt_olm3 = jet_pt / jet_pt_response_olm3
	jet_pt_olm = jet_pt / jet_pt_response_olm

	jet_pt_response_tlm1 = dataFrame['y_tlm1']
	jet_pt_response_tlm2 = dataFrame['y_tlm2']
	jet_pt_response_tlm3 = dataFrame['y_tlm3']
	jet_pt_response_tlm = dataFrame['y_tlm1'] / 3 + dataFrame['y_tlm2'] / 3 + dataFrame['y_tlm3'] / 3
	#jet_pt_response_tlm = jet_pt_response_tlm2

	jet_pt_tlm1 = jet_pt / jet_pt_response_tlm1
	jet_pt_tlm2 = jet_pt / jet_pt_response_tlm2
	jet_pt_tlm3 = jet_pt / jet_pt_response_tlm3
	jet_pt_tlm = jet_pt / jet_pt_response_tlm

	jet_pt_response_thlm1 = dataFrame['y_thlm1']
	jet_pt_response_thlm2 = dataFrame['y_thlm2']
	jet_pt_response_thlm3 = dataFrame['y_thlm3']
	jet_pt_response_thlm = dataFrame['y_thlm1'] / 3 + dataFrame['y_thlm2'] / 3 + dataFrame['y_thlm3'] / 3
	#jet_pt_response_thlm = jet_pt_response_thlm2

	jet_pt_thlm1 = jet_pt / jet_pt_response_thlm1
	jet_pt_thlm2 = jet_pt / jet_pt_response_thlm2
	jet_pt_thlm3 = jet_pt / jet_pt_response_thlm3
	jet_pt_thlm = jet_pt / jet_pt_response_thlm



	jet_pt_response_flm1 = dataFrame['y_flm1']
	jet_pt_response_flm2 = dataFrame['y_flm2']
	jet_pt_response_flm3 = dataFrame['y_flm3']
	jet_pt_response_flm = dataFrame['y_flm1'] / 3 + dataFrame['y_flm2'] / 3 + dataFrame['y_flm3'] / 3
	#jet_pt_response_flm = jet_pt_response_flm2

	jet_pt_flm1 = jet_pt / jet_pt_response_tlm1
	jet_pt_flm2 = jet_pt / jet_pt_response_tlm2
	jet_pt_flm3 = jet_pt / jet_pt_response_tlm3
	jet_pt_flm = jet_pt / jet_pt_response_tlm
	


	if scale_target == "yes":
		jet_pt_response_olm1 = dataFrame["y_olm1"] * max_jet_pt_response 
		jet_pt_response_olm2 = dataFrame["y_olm2"] * max_jet_pt_response 
		jet_pt_response_olm3 = dataFrame["y_olm3"] * max_jet_pt_response 
		jet_pt_response_olm = (dataFrame["y_olm1"] / 3 + dataFrame["y_olm2"] / 3 + dataFrame["y_olm3"] / 3) * (max_jet_pt_response - min_jet_pt_response) + min_jet_pt_response

		jet_pt_olm1 = jet_pt / jet_pt_response_olm1
		jet_pt_olm2 = jet_pt / jet_pt_response_olm2
		jet_pt_olm3 = jet_pt / jet_pt_response_olm3
		jet_pt_olm = jet_pt / jet_pt_response_olm

		jet_pt_response_tlm1 = dataFrame["y_tlm1"] * max_jet_pt_response 
		jet_pt_response_tlm2 = dataFrame["y_tlm2"] * max_jet_pt_response 
		jet_pt_response_tlm3 = dataFrame["y_tlm3"] * max_jet_pt_response 
		jet_pt_response_tlm = (dataFrame["y_tlm1"] / 3 + dataFrame["y_tlm2"] / 3 + dataFrame["y_tlm3"] / 3) * (max_jet_pt_response - min_jet_pt_response) + min_jet_pt_response

		jet_pt_tlm1 = jet_pt / jet_pt_response_tlm1
		jet_pt_tlm2 = jet_pt / jet_pt_response_tlm2
		jet_pt_tlm3 = jet_pt / jet_pt_response_tlm3
		jet_pt_tlm = jet_pt / jet_pt_response_tlm

		jet_pt_response_thlm1 = dataFrame["y_thlm1"] * max_jet_pt_response 
		jet_pt_response_thlm2 = dataFrame["y_thlm2"] * max_jet_pt_response 
		jet_pt_response_thlm3 = dataFrame["y_thlm3"] * max_jet_pt_response 
		jet_pt_response_thlm = (dataFrame["y_thlm1"] / 3 + dataFrame["y_thlm2"] / 3 + dataFrame["y_thlm3"] / 3) * (max_jet_pt_response - min_jet_pt_response) + min_jet_pt_response

		jet_pt_thlm1 = jet_pt / jet_pt_response_thlm1
		jet_pt_thlm2 = jet_pt / jet_pt_response_thlm2
		jet_pt_thlm3 = jet_pt / jet_pt_response_thlm3
		jet_pt_thlm = jet_pt / jet_pt_response_thlm

		jet_pt_response_flm1 = dataFrame["y_flm1"] * max_jet_pt_response 
		jet_pt_response_flm2 = dataFrame["y_flm2"] * max_jet_pt_response 
		jet_pt_response_flm3 = dataFrame["y_flm3"] * max_jet_pt_response
		jet_pt_response_flm = (dataFrame["y_flm1"] / 3 + dataFrame["y_flm2"] / 3 + dataFrame["y_flm3"] / 3) * (max_jet_pt_response - min_jet_pt_response) + min_jet_pt_response

		jet_pt_flm1 = jet_pt / jet_pt_response_flm1
		jet_pt_flm2 = jet_pt / jet_pt_response_flm2
		jet_pt_flm3 = jet_pt / jet_pt_response_flm3
		jet_pt_flm = jet_pt / jet_pt_response_flm


	
	for i in range(len(jet_pt_olm)):
		if jet_pt_olm[i] < 0.:
			print "pT less than 0!!! (OLM)",jet_pt_olm[i],jet_true_pt[i]
		if jet_pt_tlm[i] < 0.:
			print "pT less than 0!!! (TLM)",jet_pt_tlm[i],jet_true_pt[i]
		if jet_pt_thlm[i] < 0.:
			print "pT less than 0!!! (THLM)",jet_pt_thlm[i],jet_true_pt[i]
		if jet_pt_flm[i] < 0.:
			print "pT less than 0!!! (FLM)",jet_pt_flm[i],jet_true_pt[i]
	
	
	print "mape JES:",mape(jet_true_pt,jet_JESPt,FlatWeight)	
	print "mape olm:",mape(jet_true_pt,jet_pt_olm,FlatWeight)
	#print "mape olm1:",mape(jet_true_pt,jet_pt_olm1,FlatWeight)
	#print "mape olm2:",mape(jet_true_pt,jet_pt_olm2,FlatWeight)
	#print "mape olm3:",mape(jet_true_pt,jet_pt_olm3,FlatWeight)
	print "mape tlm:",mape(jet_true_pt,jet_pt_tlm,FlatWeight)
	#print "mape tlm1:",mape(jet_true_pt,jet_pt_tlm1,FlatWeight)
	#print "mape tlm2:",mape(jet_true_pt,jet_pt_tlm2,FlatWeight)
	#print "mape tlm3:",mape(jet_true_pt,jet_pt_tlm3,FlatWeight)
	print "mape thlm:",mape(jet_true_pt,jet_pt_thlm,FlatWeight)
	#print "mape thlm1:",mape(jet_true_pt,jet_pt_thlm1,FlatWeight)
	#print "mape thlm2:",mape(jet_true_pt,jet_pt_thlm2,FlatWeight)
	#print "mape thlm3:",mape(jet_true_pt,jet_pt_thlm3,FlatWeight)
	print "mape flm:",mape(jet_true_pt,jet_pt_flm,FlatWeight)
	#print "mape flm1:",mape(jet_true_pt,jet_pt_flm1,FlatWeight)
	#print "mape flm2:",mape(jet_true_pt,jet_pt_flm2,FlatWeight)
	#print "mape flm3:",mape(jet_true_pt,jet_pt_flm3,FlatWeight)
	
	 
	print "Define histograms"

	'''
	
	print "Fill histograms"
	for i in range(len(jet_true_pt)):
		truePtHist.Fill(jet_true_pt[i],PtWeight[i])
		predPtHistOneLayerModel.Fill(pred_jet_true_pt_OneLayerModel[i],PtWeight[i])
		predPtHistOneLayerModelDrawers.Fill(pred_jet_true_pt_OneLayerModelDrawers[i],PtWeight[i])
		predPtHistTwoLayerModel.Fill(pred_jet_true_pt_TwoLayerModel[i],PtWeight[i])
		recoPtHist.Fill(jet_pt[i],PtWeight[i])

	truePtHist.SetStats(0)
	predPtHistOneLayerModel.SetStats(0)
	predPtHistOneLayerModelDrawers.SetStats(0)
	predPtHistTwoLayerModel.SetStats(0)
	recoPtHist.SetStats(0)

	print "Make ratio"
	ratioHist = truePtHist.Clone("ratioHist")
	ratioHist.Divide(predPtHistOneLayerModel)
	ratioHist.GetYaxis().SetTitle("true / predicted")
	ratioHist.GetYaxis().SetRangeUser(0.6,1.5)

	print "Make plots"

	canvas = r.TCanvas("canvas")
	canvas.cd()
	canvas.SetLogy(True)
	truePtHist.Draw("pe")
	predPtHistOneLayerModel.Draw("pe,same")
	predPtHistOneLayerModelDrawers.Draw("pe,same")
	predPtHistTwoLayerModel.Draw("pe,same")
	canvas.Clear()

	pad1 = r.TPad("pad1","pad1",0,0.3,1,1)
	pad1.SetLogy(True)
	pad1.SetLogx(True)
	pad1.Draw()
	pad1.cd()
	truePtHist.Draw("pe")
	predPtHistOneLayerModel.Draw("pe,same")
	predPtHistOneLayerModelDrawers.Draw("pe,same")
	predPtHistTwoLayerModel.Draw("pe,same")
	recoPtHist.Draw("pe,same")
	legend = r.TLegend(0.1,0.6,0.25,0.75)
	legend.AddEntry(truePtHist,"True Spectra")
	legend.AddEntry(predPtHistOneLayerModel,"OneLayerModel")
	legend.AddEntry(predPtHistOneLayerModelDrawers,"OneLayerModelDrawers")
	legend.AddEntry(predPtHistTwoLayerModel,"TwoLayerModel")
	legend.AddEntry(recoPtHist,"Reconstructed Spectra")
	legend.SetLineWidth(0)
	legend.Draw("same")

	canvas.cd()
	pad2 = r.TPad("pad2","pad2",0,0.05,1,0.3)
	pad2.SetLogx(True)
	pad2.Draw()
	pad2.cd()
	ratioHist.Draw("pe")
	canvas.Print("ComparisonPredAndTrue.pdf")

	canvas2 = r.TCanvas("canvas2")
	canvas2.cd()
	canvas2.SetLogy(True)
	truePtHist.Draw("pe")
	predPtHistOneLayerModel.Draw("pe,same")
	canvas2.Print("ComparisonPredAndTrue_train2.pdf")

	'''


	c1 = r.TCanvas("BasicCanvas","",0, 0,800,600)

	colors = ["#ff9b54", "#c34257", "#790087", "#369ED6", "#36D69A", "#357D2C", "#BBC750"]


	jetDefsNames = []
	jetDefsRecoPts = []
	jetDefsTruePts = []
	jetDefsWeights = []

	print len(jet_pt.values), len(jet_true_pt.values), len(jet_JESPt.values), len(jet_pt_olm.values), len(jet_pt_tlm.values), len(jet_pt_thlm.values), len(jet_pt_flm.values)
	'''
	jetDefsNames.append("Reco_pT")
	jetDefsRecoPts.append(jet_pt.values)
	jetDefsTruePts.append(jet_true_pt.values)
	jetDefsWeights.append(PtWeight.values)
	'''

	
	jetDefsNames.append("JESPt")
	jetDefsRecoPts.append(jet_JESPt.values)
	jetDefsTruePts.append(jet_true_pt.values)
	jetDefsWeights.append(PtWeight.values)

	jetDefsNames.append("OneLayerModel")
	jetDefsRecoPts.append(jet_pt_olm.values)
	jetDefsTruePts.append(jet_true_pt.values)
	jetDefsWeights.append(PtWeight.values)

	jetDefsNames.append("TwoLayerModel")
	jetDefsRecoPts.append(jet_pt_tlm.values)
	jetDefsTruePts.append(jet_true_pt.values)
	jetDefsWeights.append(PtWeight.values)

	jetDefsNames.append("ThreeLayerModel")
	jetDefsRecoPts.append(jet_pt_thlm.values)
	jetDefsTruePts.append(jet_true_pt.values)
	jetDefsWeights.append(PtWeight.values)

	jetDefsNames.append("FourLayerModel")
	jetDefsRecoPts.append(jet_pt_flm.values)
	jetDefsTruePts.append(jet_true_pt.values)
	jetDefsWeights.append(PtWeight.values)

	print "Check length of pT variables:",len(jet_pt), len(jet_true_pt), len(jet_JESPt), len(jet_pt_olm), len(jet_pt_tlm), len(jet_pt_thlm), len(jet_pt_flm) 
	closureHists = []
	resolutionHists = []

	nPtBins = len(settings.ptBins)-1
	legend = r.TLegend(0.6, 0.7, 0.9, 0.9)

	closureHists = []
	resolutionHists = []

	nPtBins = len(settings.ptBins)-1
	legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
	legendFlavor = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
	legendDiff = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)

	for k in range(len(jetDefsNames)):
		responseHists = []
		responseHistsQuark = []
		responseHistsGluon = []
		responseHistsBjets = []
		name = jetDefsNames[k]
		for i in range(len(settings.ptBins)-1):
			responseHist = ROOT.TH1F("responseHist_%s_%spt_%d_%d"%(suffix, name, settings.ptBins[i], settings.ptBins[i+1]), "; p_{T}^{reco}/ p_{T}^{true}", 100, 0, 2)
			responseHist.Sumw2()
			responseHist.SetDirectory(0)
			responseHists.append(responseHist)

		jetPt = jetDefsRecoPts[k]
		ctruePt = jetDefsTruePts[k]
		ceventWeights = jetDefsWeights[k]

		closureHist = ROOT.TH1F("closure_pt_%s"%(name), ";p_{T}^{true} [GeV]; Jet p_{T} Response", nPtBins, array("d",settings.ptBins))
		resolutionHist = ROOT.TH1F("resolution_pt_%s"%(name), ";p_{T}^{true}; Jet p_{T} Resolution", nPtBins, array("d",settings.ptBins))
		closureHist.Sumw2()
		resolutionHist.Sumw2()

		closureHist.SetDirectory(0)
		resolutionHist.SetDirectory(0)

		legend.AddEntry(closureHist,name, "l")
		for j in range(len(jetPt)):
			ptBin = getPtBin(ctruePt[j])
			if ptBin < 0:
				continue
			responseHists[ptBin].Fill(jetPt[j] / ctruePt[j], ceventWeights[j])

		for i in range(len(settings.ptBins)-1):	
			#gausFit, mean, sigma, meanErr, sigmaErr = getResponseFit(responseHists[i])
			fitter = jbf.JES_BalanceFitter(1.5)
			gausFit, mean, sigma, meanErr, sigmaErr = 0, 0, 0, 0, 0
			if responseHists[i].Integral() == 0:
				print "Integral is zero!!!"
				gausFit, mean, sigma, meanErr, sigmaErr = 0, 0, 0, 0, 0
			else:
				gausFit = fitter.Fit(responseHists[i])
				mean  = gausFit.GetParameter(1)
				sigma = gausFit.GetParameter(2)
				meanErr  = gausFit.GetParError(1)
				sigmaErr = gausFit.GetParError(2)
				gausFit.SetLineColor(ROOT.kRed)
			
			closureHist.SetBinContent(i+1, mean)
			closureHist.SetBinError(i+1, meanErr)
			
			if mean > 0:
				resolutionHist.SetBinContent(i+1, sigma / mean)
				resolutionHist.SetBinError(i+1, sigmaErr / mean)
			
			if responseHists[i].Integral() != 0:
				responseHists[i].Draw()
				gausFit.Draw("same")
				saveName = "responseHist_%s_pt_%d_%d_%d.pdf"%(name, settings.ptBins[i], settings.ptBins[i+1],k)
				c1.Print(saveFolder+"/"+str(saveName))
			
		closureHists.append(closureHist)
		resolutionHists.append(resolutionHist)




	#drawHists(closureHists, "closure_%s"%(suffix), 0.8, 1.2, "", legend, colors, False)
	#drawHists(resolutionHists, "resolution_%s"%(suffix), 0.0, 0.3, "", legend, colors, False)


	closureHists[0].SetLineColor(r.kBlack)
	closureHists[1].SetLineColor(r.kRed)
	closureHists[2].SetLineColor(r.kGreen)
	closureHists[3].SetLineColor(r.kBlue)
	closureHists[4].SetLineColor(r.kMagenta)
	#closureHists[5].SetLineColor(r.kOrange)
	closureHists[0].SetMarkerColor(r.kBlack)
	closureHists[1].SetMarkerColor(r.kRed)
	closureHists[2].SetMarkerColor(r.kGreen)
	closureHists[3].SetMarkerColor(r.kBlue)
	closureHists[4].SetMarkerColor(r.kMagenta)
	#closureHists[5].SetMarkerColor(r.kOrange)
	closureHists[0].SetMarkerSize(0.5)
	closureHists[1].SetMarkerSize(0.5)
	closureHists[2].SetMarkerSize(0.5)
	closureHists[3].SetMarkerSize(0.5)
	closureHists[4].SetMarkerSize(0.5)
	#closureHists[5].SetMarkerSize(0.5)
	closureHists[0].SetStats (0)
	closureHists[1].SetStats (0)
	closureHists[2].SetStats (0)
	closureHists[3].SetStats (0)
	closureHists[4].SetStats (0)
	#closureHists[5].SetStats (0)

	resolutionHists[0].SetLineColor(r.kBlack)
	resolutionHists[1].SetLineColor(r.kRed)
	resolutionHists[2].SetLineColor(r.kGreen)
	resolutionHists[3].SetLineColor(r.kBlue)
	resolutionHists[4].SetLineColor(r.kMagenta)
	#resolutionHists[5].SetLineColor(r.kOrange)
	resolutionHists[0].SetMarkerColor(r.kBlack)
	resolutionHists[1].SetMarkerColor(r.kRed)
	resolutionHists[2].SetMarkerColor(r.kGreen)
	resolutionHists[3].SetMarkerColor(r.kBlue)
	resolutionHists[4].SetMarkerColor(r.kMagenta)
	#resolutionHists[5].SetMarkerColor(r.kOrange)
	resolutionHists[0].SetMarkerSize(0.5)
	resolutionHists[1].SetMarkerSize(0.5)
	resolutionHists[2].SetMarkerSize(0.5)
	resolutionHists[3].SetMarkerSize(0.5)
	resolutionHists[4].SetMarkerSize(0.5)
	#resolutionHists[5].SetMarkerSize(0.5)
	resolutionHists[0].SetStats (0)
	resolutionHists[1].SetStats (0)
	resolutionHists[2].SetStats (0)
	resolutionHists[3].SetStats (0)
	resolutionHists[4].SetStats (0)
	#resolutionHists[5].SetStats (0)

	closureHists[0].GetYaxis().SetRangeUser(0.85, 1.05)
	#closureHists[0].GetXaxis().SetRangeUser(200., 6000.)
	closureHists[0].GetXaxis().SetRangeUser(200., 1500.)
	#resolutionHists[0].GetYaxis().SetRangeUser(0.0, 0.09)
	#resolutionHists[0].GetXaxis().SetRangeUser(0., 0.1)
	#resolutionHists[0].GetXaxis().SetRangeUser(200., 6000.)
	resolutionHists[0].GetXaxis().SetRangeUser(200., 1500.)

	canvas_0 = r.TCanvas("canvas_0","",0, 0,800,600)
	canvas_0.cd()
	canvas_0.SetLogx(True)
	closureHists[0].Draw()
	closureHists[1].Draw("same")
	closureHists[2].Draw("same")
	closureHists[3].Draw("same")
	closureHists[4].Draw("same")
	#closureHists[5].Draw("same")
	legend_0 = r.TLegend(0.5,0.3,0.65,0.45)
	legend_0.SetTextSize(0.035)
	legend_0.SetFillStyle(0)
	legend_0.SetBorderSize(0)
	etaname = str(etaLow) + " < |#eta^{Det}| < " + str(etaHigh)
	if etaLow == 0.0:
		etaname = "|#eta^{Det}| < " + str(etaHigh)

	legend_0.AddEntry("ATLAS Internal, ",etaname)
	legend_0.AddEntry(closureHists[0],"JES")
	legend_0.AddEntry(closureHists[1],"One Layer Model")
	legend_0.AddEntry(closureHists[2],"Two Layer Model")
	legend_0.AddEntry(closureHists[3],"Three Layes Model")
	legend_0.AddEntry(closureHists[4],"Four Layer Model")
	#legend_0.AddEntry(closureHists[5],"")
	legend_0.SetLineWidth(0)
	legend_0.Draw("same")
	#astyle.ATLASLabel(0.2, 0.87, "Simulation Internal")
	#canvas_0.Update()
	canvas_0.Print(saveFolder+"/closureHists.pdf")

	canvas_1 = r.TCanvas("canvas_1","",0, 0,800,600)
	canvas_1.cd()
	canvas_1.SetLogx(True)
	resolutionHists[0].Draw()
	resolutionHists[1].Draw("same")
	resolutionHists[2].Draw("same")
	resolutionHists[3].Draw("same")
	resolutionHists[4].Draw("same")
	#resolutionHists[5].Draw("same")
	legend_1 = r.TLegend(0.5,0.6,0.65,0.75)
	legend_1.SetTextSize(0.035)
	legend_1.SetFillStyle(0)
	legend_1.SetBorderSize(0)
	legend_1.AddEntry("ATLAS Internal, ",etaname)
	legend_1.AddEntry(resolutionHists[0],"JES")
	legend_1.AddEntry(resolutionHists[1],"One Layer Model")
	legend_1.AddEntry(resolutionHists[2],"Two Layer Model")
	legend_1.AddEntry(resolutionHists[3],"Three Layes Model")
	legend_1.AddEntry(resolutionHists[4],"Four Layer Model")
	#legend_1.AddEntry(resolutionHists[5],"")
	legend_1.SetLineWidth(0)
	legend_1.Draw("same")
	#astyle.ATLASLabel(0.2, 0.87, "Simulation Internal")
	#canvas_1.Update()
	canvas_1.Print(saveFolder+"/resolutionHists.pdf")
	

parser = OptionParser()
parser.add_option('--jet_type', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
parser.add_option('--scale_target', default = "yes", type = "string", help = "scale target variable (yes or no)")
parser.add_option('--variable_type', default = "ftest", type = "string", help = "type of varible selection")
(opt, args) = parser.parse_args()
MakePlots(jet_type=opt.jet_type,etaLow = opt.etaLow,etaHigh=opt.etaHigh,scale_target=opt.scale_target,variable_type=opt.variable_type)
'''
h_true = r.TH1D("h_true",";p_{T}[GeV]; Counts, a.u.",4000, 0., 8000.)
h_reco = r.TH1D("h_true",";p_{T}[GeV]; Counts, a.u.",4000, 0., 8000.)
h_true_ptweight = r.TH1D("h_true",";p_{T}[GeV]; Counts, a.u.",4000, 0., 8000.)
h_reco_ptweight = r.TH1D("h_true",";p_{T}[GeV]; Counts, a.u.",4000, 0., 8000.)
h_true_flatweight = r.TH1D("h_true",";p_{T}[GeV]; Counts, a.u.",4000, 0., 8000.)
h_reco_flatweight = r.TH1D("h_true",";p_{T}[GeV]; Counts, a.u.",4000, 0., 8000.)

h_true.Sumw2()
h_reco.Sumw2()
h_true_ptweight.Sumw2()
h_reco_ptweight.Sumw2()
h_true_flatweight.Sumw2()
h_reco_flatweight.Sumw2()

h_true.SetLineColor(r.kRed)
h_reco.SetLineColor(r.kBlack)
h_true_ptweight.SetLineColor(r.kRed)
h_reco_ptweight.SetLineColor(r.kBlack)
h_true_flatweight.SetLineColor(r.kRed)
h_reco_flatweight.SetLineColor(r.kBlack)

h_true.SetMarkerColor(r.kRed)
h_reco.SetMarkerColor(r.kBlack)
h_true_ptweight.SetMarkerColor(r.kRed)
h_reco_ptweight.SetMarkerColor(r.kBlack)
h_true_flatweight.SetMarkerColor(r.kRed)
h_reco_flatweight.SetMarkerColor(r.kBlack)

h_true.SetMarkerSize(0.5)
h_reco.SetMarkerSize(0.5)
h_true_ptweight.SetMarkerSize(0.5)
h_reco_ptweight.SetMarkerSize(0.5)
h_true_flatweight.SetMarkerSize(0.5)
h_reco_flatweight.SetMarkerSize(0.5)

print "Fill histograms"
for i in range(len(jet_true_pt)):
	h_true.Fill(jet_true_pt[i])
	h_reco.Fill(jet_pt[i])
	h_true_ptweight.Fill(jet_true_pt[i],PtWeight[i])
	h_reco_ptweight.Fill(jet_pt[i],PtWeight[i])
	h_true_flatweight.Fill(jet_true_pt[i],FlatWeight[i])
	h_reco_flatweight.Fill(jet_pt[i],FlatWeight[i])


canvas_2 = r.TCanvas("canvas_2","",0, 0,800,600)
canvas_2.cd()
canvas_2.SetLogy(True)
canvas_2.SetLogx(True)
h_true.Draw()
h_reco.Draw("same")
legend_2 = r.TLegend(0.5,0.6,0.65,0.75)
legend_2.SetTextSize(0.035)
legend_2.SetFillStyle(0)
legend_2.SetBorderSize(0)
legend_2.AddEntry(h_true,"true")
legend_2.AddEntry(h_reco,"reconstructed")
legend_2.SetLineWidth(0)
legend_2.Draw("same")
#astyle.ATLASLabel(0.2, 0.87, "Simulation Internal")
#canvas_2.Update()
canvas_2.Print("unweighted_pt.pdf")

canvas_3 = r.TCanvas("canvas_3","",0, 0,800,600)
canvas_3.cd()
canvas_2.SetLogy(True)
canvas_3.SetLogx(True)
h_true_ptweight.Draw()
h_reco_ptweight.Draw("same")
legend_3 = r.TLegend(0.5,0.6,0.65,0.75)
legend_3.SetTextSize(0.035)
legend_3.SetFillStyle(0)
legend_3.SetBorderSize(0)
legend_3.AddEntry(h_true_ptweight,"true")
legend_3.AddEntry(h_reco_ptweight,"reconstructed")
legend_3.SetLineWidth(0)
legend_3.Draw("same")
#astyle.ATLASLabel(0.2, 0.87, "Simulation Internal")
#canvas_3.Update()
canvas_3.Print("ptweighted_pt.pdf")

canvas_4 = r.TCanvas("canvas_4","",0, 0,800,600)
canvas_4.cd()
canvas_2.SetLogy(True)
canvas_4.SetLogx(True)
h_true_flatweight.Draw()
h_reco_flatweight.Draw("same")
legend_4 = r.TLegend(0.5,0.6,0.65,0.75)
legend_4.SetTextSize(0.035)
legend_4.SetFillStyle(0)
legend_4.SetBorderSize(0)
legend_4.AddEntry(h_true_flatweight,"true")
legend_4.AddEntry(h_reco_flatweight,"reconstructed")
legend_4.SetLineWidth(0)
legend_4.Draw("same")
#astyle.ATLASLabel(0.2, 0.87, "Simulation Internal")
#canvas_4.Update()
canvas_4.Print("flatweighted_pt.pdf")
'''
