import numpy as np
from sklearn import linear_model,neural_network
from numpy import random
from root_numpy import testdata
import pdb
import ROOT
from keras.models import Sequential
from keras.layers import Dense
import h5py
import settings as settings
import sys
import pickle
from genNumericalInversion import genNumericalInverter
from sklearn import preprocessing
import keras.backend as K
from tensorflow.python.ops import math_ops
import JES_BalanceFitter as jbf

import AtlasStyle as AS


def msape(y_true, y_pred):
  return K.mean(np.divide((y_pred-y_true), y_true)*np.divide((y_pred-y_true), y_true), axis=-1)
  #return abs(K.mean(np.divide((y_pred), y_true)*np.divide((y_pred), y_true), axis=-1)-1)



def getBinIndex(value, bins):
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


def draw_atlas_details(labels=[],x_pos= 0.2,y_pos = 0.87, dy = 0.04*0.9, text_size = 0.035*0.9, sampleName="", height=1.0, isSimulation = True):
    if sampleName != "":
          sampleName = ", " + c.samples[sampleName]["Name"]
    text_size = text_size / height
    dy = dy / height
    if not isSimulation:
      AS.ATLASLabel(  x_pos, y_pos,1, 0.1, text_size,"Internal")
    else:
      AS.ATLASLabel(  x_pos, y_pos,1, 0.1, text_size,"Simulation Internal")
    y_pos -= dy
    AS.myText(  x_pos, y_pos,1,text_size,"#sqrt{s} = 13 TeV %s"%(sampleName))
    y_pos -= dy

    for label in labels:
        AS.myText( x_pos, y_pos, 1, text_size, "%s"%label)
        y_pos -= dy


def shuffleData(truePt, recoPt, eventWeights, features, truthLabel):
    shuffled_true = np.empty(truePt.shape, dtype=truePt.dtype)
    shuffled_reco = np.empty(recoPt.shape, dtype=recoPt.dtype)
    shuffled_weight = np.empty(eventWeights.shape, dtype=eventWeights.dtype)
    shuffled_feat = np.empty(features.shape, dtype=features.dtype)
    shuffled_label = np.empty(truthLabel.shape, dtype=truthLabel.dtype)
    permutation = np.random.permutation(len(truePt))

    for old_index, new_index in enumerate(permutation):
        shuffled_true[new_index] = truePt[old_index]
        shuffled_reco[new_index] = recoPt[old_index]
        shuffled_weight[new_index] = eventWeights[old_index]
        shuffled_label[new_index] = truthLabel[old_index]
        for i in range(len(features[old_index])):
          shuffled_feat[new_index][i] = features[old_index][i]

    return shuffled_true, shuffled_reco, shuffled_weight, shuffled_feat, truthLabel

def normalizeFeature(feature):

  #minVal = min(feature)
  #maxVal = max(feature)
  #newFeature = (feature-minVal) / (maxVal - minVal) - 0.5

  scaler = preprocessing.StandardScaler().fit(feature)
  #scaler = preprocessing.MinMaxScaler().fit(feature)
  #https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
  #scaler = preprocessing.RobustScaler().fit(feature)
  newFeature = scaler.transform(feature)
  return newFeature

def getNbins(variables):
  nbins = 1
   
  for j in range(len(variables)):
     var = variables[j]
     binning = settings.testBins[var]
     nbins = nbins * len(binning)

  return nbins

def getSuffix(etaBin):
  suffix = settings.fullEventWeight + "_" + settings.loss + "_" + "etaBin_" + str(etaBin)
  #suffix = settings.fullEventWeight + "_" + settings.loss 
  for feature in settings.trainingFeatures:
    suffix = suffix + "_" + feature
  return suffix



def getRebin(variableLengths, bins):
  if len(variableLengths) == 1:
    return bins[0]

  nBins = len(variableLengths)
  #tmpBin = bins[nBins-1]*variableLengths[nBins-2] + bins[nBins-2]
  tmpBin = bins[nBins-2]*variableLengths[nBins-1] + bins[nBins-1]
  bins[nBins-2] = tmpBin
  variableLengths[nBins-2] = variableLengths[nBins-1] * variableLengths[nBins-2]

  variableLengths.pop(nBins-1)
  bins.pop(nBins-1)

  return getRebin(variableLengths, bins)

def getBin(variableNames, values):
  bins = []
  binLengths = []
  
  for j in range(len(variableNames)):
     var = variableNames[j]
     binning = settings.testBins[var]
     binLengths.append( len(binning))

     for i in range(0, len(binning)-1):
       if values[j] < binning[i]: 
         bins.append(i-1)
         break
     if len(bins) != len(binLengths):
       bins.append(i-1)

  newBin = getRebin(binLengths, bins)
  return newBin
     
     
def unNormalizeFeature(feature, normFeature):
  minVal = min(feature)
  maxVal = max(feature)
  newFeature = (normFeature + 0.5)*(maxVal - minVal) + minVal 

  return newFeature


################################################
## Get the data in a readable format
################################################

def getInputs(etaBin, doShuffle, doNorm = True, infile = settings.hdf5FileName, featureNames = settings.trainingFeatures, isShuffled = False):

  if(isShuffled) :
    infile ="shuffled_"+infile 
  infileWeights = h5py.File("hdf5Files/"+infile + "_" + settings.fullEventWeight + "_etaBin_" + str(etaBin) + '.hdf5', 'r')
  eventWeights = infileWeights[settings.fullEventWeight]


  infileTruePt = h5py.File("hdf5Files/"+infile + "_" + "jet_true_pt" + "_etaBin_" + str(etaBin) + '.hdf5', 'r')
  truePtTmp = infileTruePt['jet_true_pt']
  truePt = np.asarray(truePtTmp, dtype=np.float32)
  #truePt = normalizeFeature(truePt)
  if(doNorm):
    truePt = truePt /  settings.maxJetPt

  infileRecoPt = h5py.File("hdf5Files/"+infile + "_" + "jet_JESPt" + "_etaBin_" + str(etaBin) + '.hdf5', 'r')
  recoPtJESTmp = infileRecoPt['jet_JESPt']
  recoPtJES = np.asarray(recoPtJESTmp, dtype=np.float32)
  #recoPtJES = normalizeFeature(recoPtJES)
  if(doNorm):
    recoPtJES = recoPtJES / settings.maxJetPt
  #recoPtJES = (recoPtJES / settings.maxJetPt) / truePt

  infileTruthLabel = h5py.File("hdf5Files/"+infile + "_" + settings.truthLabel + "_etaBin_" + str(etaBin) + '.hdf5', 'r')
  truthLabelTmp = infileTruthLabel[settings.truthLabel]
  truthLabel = np.asarray(truthLabelTmp, dtype=np.int32)

  featuresArr = []
  for featureName in featureNames:
    infileFeature = h5py.File("hdf5Files/"+infile + "_" + featureName + "_etaBin_" + str(etaBin)+".hdf5", 'r')
    featureTmp = infileFeature[featureName]
    feature = np.asarray(featureTmp, dtype=np.float32).reshape(-1,1)
    if doNorm:
       feature = normalizeFeature(feature)
    featuresArr.append(feature)

  fullFeatures = np.hstack(featuresArr)
  if doShuffle and not isShuffled:
    truePt, recoPtJES, eventWeights, fullFeatures, truthLabel = shuffleData(truePt, recoPtJES, eventWeights, fullFeatures, truthLabel)

  return truePt, recoPtJES, eventWeights, fullFeatures, truthLabel


