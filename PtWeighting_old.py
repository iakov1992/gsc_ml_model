import ROOT as r 
import h5py
import numpy as np
from optparse import OptionParser


def construct_pt_weighting(samples, ptVariable, ptVarReco, minPt, maxPt, eventWeight = "mcEventWeight", outName = "FullWeight"):
	print "retrieving truePt"
	ptHist = r.TH1D("ptHist", "ptHist; pT", 1000, 0., 4000.)
	eventWeightsHist = r.TH1D("eventWeightsHist", "eventWeightsHist; eventWeight", 10000, 0., 400.)
	eventWeightPtCorr = r.TH2D("eventWeightPtCorr","eventWeightPtCorr; pT; eventWeight",1000, 0., 4000.,10000, 0., 400.)

	for sample in samples:
		infileTruePt = h5py.File("hdf5Files/mc_sample_" + sample + "_0_" + ptVariable + '.hdf5', 'r')
		truePtTmp = infileTruePt[ptVariable]
		truePt = np.asarray(truePtTmp, dtype=np.float32)

		infileWeights = h5py.File("hdf5Files/mc_sample_" + sample + "_0_" + eventWeight + '.hdf5', 'r')
		print "retrieving event weights for ", sample
		eventWeights = np.asarray(infileWeights[eventWeight], dtype=np.float32)

		infileRecoPt = h5py.File("hdf5Files/mc_sample_"+ sample + "_0_" + ptVarReco + '.hdf5', 'r')
		recoPtTmp = infileRecoPt[ptVarReco]
		recoPt = np.asarray(recoPtTmp, dtype=np.float32)

		for j in range(len(truePt)):
			ptHist.Fill(truePt[j], eventWeights[j]);
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

	c1 = r.TCanvas("pTHist","pTHist",800,600)
	c1.cd()
	c1.SetLogx(1)
	c1.SetLogy(1)
	ptHist.Draw()
	c1.Print("ptHist.pdf")

	c2 = r.TCanvas("flatHist","flatHist",800,600)
	c2.cd()
	c2.SetLogx(1)
	flatHist.Draw()
	c2.Print("flatHist.pdf")



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

	c3 = r.TCanvas("weightHist","weightHist",800,600)
	c3.cd()
	c3.SetLogx(1)
	c3.SetLogy(1)
	weightHist.Draw()
	c3.Print("weightHist.pdf")


	c4 = r.TCanvas("eventWeightsHist","eventWeightsHist",800,600)
	c4.cd()
	c4.SetLogx(1)
	c4.SetLogy(1)
	eventWeightsHist.Draw()
	c4.Print("eventWeightsHist.pdf")

	c5 = r.TCanvas("eventWeightPtCorr","eventWeightPtCorr",800,600)
	c5.cd()
	c5.SetLogx(1)
	c5.SetLogy(1)
	c5.SetLogz(1)
	eventWeightPtCorr.Draw("colz")
	c5.Print("eventWeightPtCorr.pdf")

	newWeightHist = r.TH1D("newWeightHist", "newWeightHist; newWeightHist", 1000000, 0., 1.e-2)

	for sample in samples:
		infileTruePt = h5py.File("hdf5Files/mc_sample_" + sample + "_0_" + ptVariable + '.hdf5', 'r')
		truePtTmp = infileTruePt[ptVariable]
		truePt = np.asarray(truePtTmp, dtype=np.float32)

		infileWeights = h5py.File("hdf5Files/mc_sample_" + sample + "_0_" + eventWeight + '.hdf5', 'r')
		print "retrieving event weights for ", sample
		eventWeights = np.asarray(infileWeights[eventWeight], dtype=np.float32)

		infileRecoPt = h5py.File("hdf5Files/mc_sample_"+ sample + "_0_" + ptVarReco + '.hdf5', 'r')
		recoPtTmp = infileRecoPt[ptVarReco]
		recoPt = np.asarray(recoPtTmp, dtype=np.float32)

		noweightArr = []
    	weightArr = []

        for j in range(len(truePt)):
			if(recoPt[j] < minPt):
				print "recoPt[j]  = ", recoPt[j], "< minPt = ", minPt
				weightArr.append(0)
				noweightArr.append(1)
				continue
			newWeight = eventWeights[j] * weightHist.GetBinContent( ptHist.FindBin(truePt[j]) )
			if newWeight > 0:
				print newWeight
			#print "eventWeights[j]  = ", eventWeights[j], "weightHist.GetBinContent( ptHist.FindBin(truePt[j]) ) = ", weightHist.GetBinContent( ptHist.FindBin(truePt[j]) )
			weightArr.append( newWeight )
			noweightArr.append(1)
			newWeightHist.Fill(newWeight)

	c6 = r.TCanvas("newWeightHist","newWeightHist",800,600)
	c6.cd()
	c6.SetLogx(1)
	c6.SetLogy(1)
	newWeightHist.Draw()
	c6.Print("newWeightHist.pdf")

	print "newWeightHist.Integral() = ", newWeightHist.Integral()
'''
        #print weightArr
        outfile = h5py.File("hdf5Files/mc_sample_"+ sample + "_0_"+ outName  + ".hdf5", 'w')
        dset = outfile.create_dataset(outName, data=weightArr, maxshape=(None,))
        outNameNoWeight = "NoWeight"
        outfile = h5py.File("hdf5Files/mc_sample_" + sample + "_0_"+ outNameNoWeight + ".hdf5", 'w')
        dset = outfile.create_dataset(outNameNoWeight, data=noweightArr, maxshape=(None,))
'''

parser = OptionParser()
parser.add_option('--ptVariable', default = "jet_true_pt")
parser.add_option('--ptVarReco', default = "jet_pt")
parser.add_option('--eventWeight', default = "weight")
parser.add_option('--outName', default = "PtWeight")
parser.add_option('--minPtReco', default = 20, type="int")
parser.add_option('--maxPtReco', default = 4000, type="int")
parser.add_option('--samples', action="append", type="string")

(opt, args) = parser.parse_args()
if not opt.samples:
	print "can't find the sample!!!"

construct_pt_weighting(opt.samples, opt.ptVariable, opt.ptVarReco, opt.minPtReco, opt.maxPtReco, eventWeight=opt.eventWeight, outName = opt.outName)





