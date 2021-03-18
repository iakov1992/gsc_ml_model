## Relevenat information for creating hdf5 files
#fileList = "fileList.txt"
#hdf5FileName = "fullDataFile"
#fileList = "sherpaFiles.txt"
#hdf5FileName = "sherpaFile"
#hdf5GenNiFileName = "fileFile"
#fileList = "jz1List.txt"
#hdf5FileName = "jz1File"
#fileList = "sherpaJZ1.txt"
#hdf5FileName = "sherpaJZ1"
hdf5SherpaFileName = "sherpaJZ1"
fileList = "shortFileList.txt"
hdf5FileName = "shortFile"

treeName = "IsolatedJet_tree"
selection='jet_true_pt > 0'
#eventWeight = "weight"
eventWeight = "mcEventWeight"
fullEventWeight = "FullWeight"
#fullEventWeight = "weight"
#fullEventWeight = "mcEventWeight"
#fullEventWeight = "NoWeight"
etaBins = [0.2, 0.7]
#etaBins = [0.0, 0.2, 0.7, 1.3, 1.8, 2.5, 2.8, 3.2, 3.5, 4.5]
minTruthPt = 10
maxJetPt = 500
nJets = 2
#maxJetPt = 60
validationSplit = 0.1
maxEvents = 10000000
doRatio = True
doNorm = True

# The standard features
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "EM3", "TILE0"]
# The standard features plus detector eta
trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "jet_ChargedFraction", "EM3", "TILE0", "jet_DetEta"]
#
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "EM3", "TILE0", "jet_DetEta", "jet_trk_c1beta02"]
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "EM3", "TILE0", "jet_DetEta", "jet_PartonTruthLabelID"]
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "TILE0", "EM3",  "jet_nMuSeg"]
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "EM3", "TILE0", "jet_trk_nca", "jet_trk_ncasd", "jet_trk_rg", "jet_trk_zg", "jet_trk_c1beta02", "jet_DetEta"]
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "jet_DetEta", "EM0", "EM1", "EM2", "EM3", "TILE0", "TILE1", "TILE2", "jet_ChargedFraction"]
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "EM3", "TILE0", "jet_DetEta", "jet_trk_c1beta02"]
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "EM3", "TILE0", "jet_DetEta", "jet_PartonTruthLabelID", "EM0", "EM1", "EM2"]
#trainingFeatures = ["TILE0", "EM3"]
#trainingFeatures = ["jet_Wtrk1000", "jet_Ntrk1000"]

CorrelationFeatures = ["jet_Wtrk1000", "jet_Ntrk1000", "EM0", "EM1", "EM2", "EM3", "TILE0", "TILE1", "TILE2", "jet_trk_nca", "jet_trk_ncasd", "jet_trk_rg", "jet_trk_zg", "jet_trk_c1beta02", "jet_DetEta"]

loss = "mse"

etaName = "jet_DetEta"
truePtName = "jet_true_pt"

ptNames = ["jet_pt", "jet_JESPt", "jet_true_pt"]
#extrasList = ["jet_Wtrk1000", "jet_Ntrk1000",  "jet_nMuSeg", "jet_PartonTruthLabelID"]
#extrasList = ["jet_Wtrk1000"]
extrasList = ["jet_PartonTruthLabelID", "jet_trk_nca", "jet_trk_ncasd", "jet_trk_rg", "jet_trk_zg", "jet_trk_c1beta02", "jet_Wtrk1000", "jet_Ntrk1000", "jet_DetEta", "jet_ChargedFraction"]
#extrasList = ["jet_trk_nca", "jet_trk_ncasd", "jet_trk_rg", "jet_trk_zg", "jet_trk_c1beta02"]
#extrasList = ["jet_PartonTruthLabelID", "jet_Wtrk1000", "jet_Ntrk1000", "jet_DetEta"]
#extrasList = ["jet_PartonTruthLabelID"]
#extrasList = ["jet_Wtrk1000", "jet_Ntrk1000"]
vectorFeatureName = "jet_EnergyPerSampling"
vectorFeaturesList = {}
vectorFeaturesList["EM0"] = [0, 4]
vectorFeaturesList["EM1"] = [1, 5]
vectorFeaturesList["EM2"] = [2, 6]
vectorFeaturesList["EM3"] = [3, 7]
vectorFeaturesList["TILE0"] = [12, 18]
vectorFeaturesList["TILE1"] = [13, 19]
vectorFeaturesList["TILE2"] = [14, 20]


## 
pickleFileName = "gNI"

truthLabel = "jet_PartonTruthLabelID"

#extraFeatures = ["jet_Wtrk1000", "jet_Ntrk1000",  "jet_nMuSeg", "jet_PartonTruthLabelID", "EM3", "TILE0", "jet_true_pt"]
extraFeatures = ["jet_Wtrk1000", "jet_Ntrk1000",  "EM3", "TILE0"]


outputDirectory = "Results"

#ptBins = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800]
#ptBins = [20, 25, 30, 35, 40, 50]
#ptBins = [20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300, 400, 500, 600, ]
ptBins = [200., 250., 300., 350., 400., 500., 600., 800., 1000., 1200., 1500., 1800., 2100., 2500., 3500., 4500.,6000.,8000.]
#ptBins = [20, 25, 30, 35, 40, 45, 50, 55, 60]
#ptBins = [20, 30, 40, 50, 60]

testBins = {}
testBins["jet_Ntrk1000"] = [-0.001, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30]
#testBins["jet_Ntrk1000"] = [-0.001, 2, 4, 6, 8, 10]
testBins["jet_Wtrk1000"] = [0, 0.10, 0.20, 0.30, 0.4]
testBins["EM3"] = [-0.025, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
#testBins["EM3"] = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
testBins["TILE0"] = [-0.1, 0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.6]
#testBins["jet_true_pt"] = [15, 65, 115, 165, 215]
testBins["TILE0"] = [-0.1, 0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6]
#testBins["jet_nMuSeg"] = [0, 10, 40, 60, 80, 100, 150, 200, 300, 400]


weightBins = {}
weightBins["jet_Ntrk1000"] = [30, -1, 30]
weightBins["EM3"] = [30, -0.05, 0.1]
weightBins["jet_Wtrk1000"] = [20, 0, 0.4]
weightBins["TILE0"] = [10, -0.1, 0.6]
weightBins["jet_true_pt"] = [500, 0, 2000]
weightBins["jet_nMuSeg"] = [10, 0, 20]



