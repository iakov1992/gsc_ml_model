import ROOT as r 
import h5py
import numpy as np
from optparse import OptionParser


def createH5File(variable, sample, outName, etaLow, etaHigh, jet_type):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
	outfile = h5py.File(folder_name + "mc_sample_"+ str(sample) + "_"+ str(outName)  + ".hdf5", 'w')
	dset = outfile.create_dataset(outName, data=variable, maxshape=(None,))
	outfile.close()

def createWeightHist(samples, ptVariable, ptVarReco, minPt, maxPt, etaLow, etaHigh, jet_type, eventWeight = "mcEventWeight"):
	print "retrieving truePt"
	ptHist = r.TH1D("ptHist", "ptHist; pT", 4000, 0., 8000.)
	eventWeightsHist = r.TH1D("eventWeightsHist", "eventWeightsHist; eventWeight", 10000, 0., 400.)
	eventWeightPtCorr = r.TH2D("eventWeightPtCorr","eventWeightPtCorr; pT; eventWeight",4000, 0., 8000.,10000, 0., 400.)
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
	
	for sample in samples:
		infileTruePt = h5py.File(folder_name + "mc_sample_" + str(sample) + "_" + str(ptVariable) + '.hdf5', 'r')
		truePtTmp = infileTruePt[ptVariable]
		truePt = np.asarray(truePtTmp, dtype=np.float32)
		
		infileWeights = h5py.File(folder_name + "mc_sample_" + str(sample) + "_" + str(eventWeight) + '.hdf5', 'r')
		print "retrieving event weights for ", sample
		eventWeights = np.asarray(infileWeights[eventWeight], dtype=np.float32)
		
		infileRecoPt = h5py.File(folder_name + "mc_sample_"+ str(sample) + "_" + str(ptVarReco) + '.hdf5', 'r')
		recoPtTmp = infileRecoPt[ptVarReco]
		recoPt = np.asarray(recoPtTmp, dtype=np.float32)
		
		for j in range(len(truePt)):
			ptHist.Fill(truePt[j], eventWeights[j])
			if eventWeights[j] < 0:
				print "truePt[j] = ", truePt[j], ", eventWeights[j]  = ",eventWeights[j]
			if eventWeights[j] > 400:
				print "truePt[j] = ", truePt[j], ", eventWeights[j]  = ",eventWeights[j]
			eventWeightsHist.Fill(eventWeights[j])
			eventWeightPtCorr.Fill(truePt[j],eventWeights[j])
			
	print "done filling pt histogram"
	flatHist = ptHist.Clone("FlatPtHist")
	weightHist = ptHist.Clone("WeightHist")
	weightHist.Reset()
	
	for i in range(flatHist.GetNbinsX()):
		flatHist.SetBinContent(i+1, 1)
		flatHist.SetBinError(i+1,0)
		
	print "ptHist.Integral()", ptHist.Integral()
	ptHist.Scale(1. / ptHist.Integral())
	print "flatHist.Integral()", flatHist.Integral()
	flatHist.Scale(1. / flatHist.Integral())
	
	print "ptHist.GetNbinsX()", ptHist.GetNbinsX()
	for cbin in range(0, ptHist.GetNbinsX()):
		print "ptHist.GetBinCenter(",cbin+1,") = ",ptHist.GetBinCenter(cbin + 1), "ptHist.GetBinContent(",cbin+1,") = ",ptHist.GetBinContent(cbin + 1), "flatHist.GetBinContent(",cbin + 1,") =", flatHist.GetBinContent(cbin + 1)
		#print ptHist.GetBinContent(cbin + 1) > 0
		weight = 0
		if ptHist.GetBinContent(cbin + 1) > 0:
			weight =  flatHist.GetBinContent(cbin + 1) / ptHist.GetBinContent(cbin + 1)
			weightHist.SetBinContent(cbin+1, weight)
			weightHist.SetBinError(cbin+1,0)
			print weight
			
	#print "Done filling full weight array"
	#print "Finalizing the weights, getting rid of outliers"
	 
	return weightHist, ptHist

def construct_pt_weighting(sample,samples, ptVariable, ptVarReco, minPt, maxPt, etaLow, etaHigh, jet_type, eventWeight = "mcEventWeight", outName = "FullWeight"):
	print "sample = ", sample
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)+"/"
	
	infileTruePt = h5py.File(folder_name + "mc_sample_" + str(sample) + "_" + str(ptVariable) + '.hdf5', 'r')
	truePtTmp = infileTruePt[ptVariable]
	truePt = np.asarray(truePtTmp, dtype=np.float32)
	
	infileWeights = h5py.File(folder_name + "mc_sample_" + str(sample) + "_" + str(eventWeight) + '.hdf5', 'r')
	print "retrieving event weights for ", sample
	eventWeights = np.asarray(infileWeights[eventWeight], dtype=np.float32)
	
	infileRecoPt = h5py.File(folder_name+"mc_sample_"+ str(sample) + "_" + str(ptVarReco) + '.hdf5', 'r')
	recoPtTmp = infileRecoPt[ptVarReco]
	recoPt = np.asarray(recoPtTmp, dtype=np.float32)
	
	noweightArr = []
	weightArr = []
	weightHist, ptHist = createWeightHist(samples, ptVariable, ptVarReco, minPt, maxPt, etaLow, etaHigh, jet_type, eventWeight)
	print "weightHist.Integral()", weightHist.Integral(), "ptHist.Integral()", ptHist.Integral()
	
	for j in range(len(truePt)):
		if(recoPt[j] < minPt):
			print "recoPt[j]  = ", recoPt[j], "< minPt = ", minPt
			weightArr.append(0)
			noweightArr.append(1)
			continue
		newWeight = eventWeights[j] * weightHist.GetBinContent( ptHist.FindBin(truePt[j]) )
		weightArr.append( newWeight )
		noweightArr.append(1)
		
	outfile = h5py.File(folder_name + "mc_sample_"+ str(sample) + "_"+ str(outName)  + ".hdf5", 'w')
	dset = outfile.create_dataset(outName, data=weightArr, maxshape=(None,))



parser = OptionParser()
parser.add_option('--ptVariable', default = "jet_true_pt")
parser.add_option('--ptVarReco', default = "jet_pt")
parser.add_option('--eventWeight', default = "weight")
parser.add_option('--outName', default = "PtWeight")
parser.add_option('--minPtReco', default = 0, type="int")
parser.add_option('--maxPtReco', default = 8000, type="int")
parser.add_option('--samples', action="append", type="string")
parser.add_option('--sample', default="364701", type="string")
parser.add_option('--jettype', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")

(opt, args) = parser.parse_args()
if not opt.sample:
	print "can't find the sample!!!"

construct_pt_weighting(opt.sample,opt.samples, opt.ptVariable, opt.ptVarReco, opt.minPtReco, opt.maxPtReco, etaLow=opt.etaLow,etaHigh=opt.etaHigh,jet_type=opt.jettype, eventWeight=opt.eventWeight, outName = opt.outName)

