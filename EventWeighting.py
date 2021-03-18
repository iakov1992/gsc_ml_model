import h5py
import numpy as np
from optparse import OptionParser


def constructEventWeighting(sample,eventWeight,outName,etaLow,etaHigh,jet_type):
	folder_name = "jets"
	if jet_type == "LCTopo":
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)
	else:
		folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)
		
	infile = folder_name + "/mc_sample_" + str(sample) + "_" + eventWeight + ".hdf5"
	print infile
	infileWeights = h5py.File(infile, "r")
	eventWeights = infileWeights[eventWeight].value
	
	infile = folder_name + "/mc_sample_" + str(sample) + "_" + "SumWeights" + ".hdf5"
	infileSumWeights = h5py.File(infile, "r")
	print infile
	eventSumWeights = infileSumWeights['SumWeights'].value
	
	infile = folder_name + "/mc_sample_" + str(sample) + "_" + "EventCount" + ".hdf5"
	infileNEvents = h5py.File(infile, "r")
	print infile
	neventWeights = infileNEvents['EventCount'].value
	
	weightArr = []
	
	scaleFactor = 0
	for j in range(len(eventWeights)):
		scaleFactor = 1. / eventSumWeights[j]
		weightArr.append(eventWeights[j] * scaleFactor)
		
	outfile = h5py.File(folder_name + "/mc_sample_" + sample + "_" + outName + ".hdf5", 'w')
	dset = outfile.create_dataset(outName, data=weightArr, maxshape=(None,))
	outfile.close()


parser = OptionParser()
parser.add_option('--samples', action="append", type="string")
parser.add_option('--eventWeight', default = "weight")
parser.add_option('--outName', default = "PtWeight")
parser.add_option('--jettype', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
(opt, args) = parser.parse_args()
if not opt.samples:
	print "Could not find samples, using the default in settings file"

for sam in opt.samples:
	constructEventWeighting(sample=sam,eventWeight=opt.eventWeight,outName = opt.outName,etaLow=opt.etaLow,etaHigh=opt.etaHigh,jet_type=opt.jettype)

