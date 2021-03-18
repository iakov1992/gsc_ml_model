import pandas as pd
import numpy as np
import scipy
from root_numpy import tree2array
import ROOT
import h5py
from optparse import OptionParser
import os


def createh5files(filename,variableName,variable,etaLow,etaHigh,jet_type):
    folder_name = "jets"
    if jet_type == "LCTopo":
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)
    output_name = folder_name + "/" +filename+"_"+variableName+".hdf5"
    if os.path.exists(output_name):
        print "File",output_name,"already exist! Remove the file to create a new one"
        os.remove(output_name)
    outFile = h5py.File(output_name,"w")
    dset = outFile.create_dataset(variableName, data=variable, maxshape=(None,))
    outFile.close()

def createDataset(filename, treename, sample, etaLow, etaHigh,jet_type):
    print "Processing the root file: ", filename, ", tree name: ", treename
    folder_name = "jets"
    if jet_type == "LCTopo":
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)
    else:
        folder_name = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/hdf5Files/etaLow_"+str(etaLow)+"_etaHigh_"+str(etaHigh)
    print "Create folder:"
    if os.path.exists(folder_name):
        print "Folder",folder_name,"already exists"
    else:
        os.mkdir(folder_name)
    n_iter = 0
    rfile  = ROOT.TFile.Open(filename)
    intree = rfile.Get(treename)   
    
    runNumber       = []
    eventNumber     = []
    lumiBlock = []
    bcid = []
    rhoEM = []
    mcEventWeight   = []
    NPV             = []
    actualInteractionsPerCrossing = []
    averageInteractionsPerCrossing = []
    #weight_pileup   = []
    weight          = []
    rho             = []
    #weight_xs       = []
    #weight_mcEventWeight = []
    njet            = []
    jet_E           = []
    jet_pt          = []
    jet_phi         = []
    jet_eta         = []
    jet_DetEta      = []
    jet_Jvt         = []
    jet_true_pt     = []
    jet_true_eta    = []
    jet_true_phi    = []
    jet_true_e      = []
    jet_respE       = []
    jet_respPt      = []
    jet_PartonTruthLabelID = []
    jet_Wtrk1000    = []
    jet_Ntrk1000    = []
    jet_EnergyPerSampling_0 = []
    jet_EnergyPerSampling_1 = []
    jet_EnergyPerSampling_2 = []
    jet_EnergyPerSampling_3 = []
    jet_EnergyPerSampling_4 = []
    jet_EnergyPerSampling_5 = []
    jet_EnergyPerSampling_6 = []
    jet_EnergyPerSampling_7 = []
    jet_EnergyPerSampling_8 = []
    jet_EnergyPerSampling_9 = []
    jet_EnergyPerSampling_10 = []
    jet_EnergyPerSampling_11 = []
    jet_EnergyPerSampling_12 = []
    jet_EnergyPerSampling_13 = []
    jet_EnergyPerSampling_14 = []
    jet_EnergyPerSampling_15 = []
    jet_EnergyPerSampling_16 = []
    jet_EnergyPerSampling_17 = []
    jet_EnergyPerSampling_18 = []
    jet_EnergyPerSampling_19 = []
    jet_EnergyPerSampling_20 = []
    jet_EnergyPerSampling_21 = []
    jet_EnergyPerSampling_22 = []
    jet_EnergyPerSampling_23 = []
    jet_EnergyPerSampling_24 = []
    jet_EnergyPerSampling_25 = []
    jet_EnergyPerSampling_26 = []
    jet_EnergyPerSampling_27 = []
    jet_ConstitE    = []
    jet_ChargedFraction = []
    jet_nMuSeg      = []
    jet_trk_isd     = []
    jet_trk_nca     = []
    jet_trk_ncasd   = []
    jet_trk_rg      = []
    jet_trk_zg      = []
    jet_trk_c1beta02 = []
    #rhoEMPFLOW = []
    jet_GhostArea = []
    jet_ActiveArea = []
    jet_VoronoiArea = []
    jet_ActiveArea4vec_pt = []
    jet_ActiveArea4vec_eta  = []
    jet_ActiveArea4vec_phi  = []
    jet_ActiveArea4vec_m  = []
    jet_n90Constituents = []
    jet_m = []           
    jet_true_m = []      
    jet_Ntrk500 = []     
    jet_Wtrk500 = []    
    jet_Ntrk2000 = []    
    jet_Ntrk3000 = []    
    jet_Ntrk4000 = []    
    jet_Wtrk2000 = []    
    jet_Wtrk3000 = []    
    jet_Wtrk4000 = []    
    jet_ConstitPt = []   
    jet_ConstitEta = []  
    jet_ConstitPhi = []  
    jet_ConstitMass = [] 
    #jet_PileupE = []     
    #jet_PileupPt = []    
    #jet_PileupEta = []   
    #jet_PileupPhi = []   
    #jet_PileupMass = []  
    #jet_EME = []        
    #jet_EMPt = []        
    #jet_EMEta = []       
    #jet_EMPhi = []       
    #jet_EMMass = []      
    jet_JESE = []        
    jet_JESPt = []       
    jet_JESEta = []      
    jet_JESPhi = []      
    jet_JESMass = []     
    jet_iso_dR = []      
    jet_ghostFrac = []
    
    histName = "MetaData_EventCount"
    branchHist = rfile.Get(histName)
    
    EventCount = [] 
    SumWeights = [] 
    
    for i in range(0, intree.GetEntries()):
        intree.GetEntry(i)
        if i % 100000 == 0:
            print ("running event ", i)
        runNumber_tmp       = getattr(intree,"runNumber")
        eventNumber_tmp     = getattr(intree,"eventNumber")
        lumiBlock_tmp = getattr(intree,"lumiBlock")
        bcid_tmp = getattr(intree,"bcid")
        rhoEM_tmp = getattr(intree,"rhoEM")
        mcEventWeight_tmp   = getattr(intree,"mcEventWeight")
        NPV_tmp             = getattr(intree,"NPV")
        actualInteractionsPerCrossing_tmp = getattr(intree,"actualInteractionsPerCrossing")
        averageInteractionsPerCrossing_tmp = getattr(intree,"averageInteractionsPerCrossing")
        #weight_pileup_tmp   = getattr(intree,"weight_pileup")
        weight_tmp          = getattr(intree,"weight")
        rho_tmp             = getattr(intree,"rho")
        #weight_xs_tmp       = getattr(intree,"weight_xs")
        #weight_mcEventWeight_tmp = getattr(intree,"weight_mcEventWeight")
        njet_tmp            = getattr(intree,"njet")
        jet_E_tmp           = getattr(intree,"jet_E")
        jet_pt_tmp          = getattr(intree,"jet_pt")
        jet_phi_tmp         = getattr(intree,"jet_phi")
        jet_eta_tmp         = getattr(intree,"jet_eta")
        jet_DetEta_tmp      = getattr(intree,"jet_DetEta")
        jet_Jvt_tmp         = getattr(intree,"jet_Jvt")
        jet_true_pt_tmp     = getattr(intree,"jet_true_pt")
        jet_true_eta_tmp    = getattr(intree,"jet_true_eta")
        jet_true_phi_tmp    = getattr(intree,"jet_true_phi")
        jet_true_e_tmp      = getattr(intree,"jet_true_e")
        jet_respE_tmp       = getattr(intree,"jet_respE")
        jet_respPt_tmp      = getattr(intree,"jet_respPt")
        jet_PartonTruthLabelID_tmp = getattr(intree,"jet_PartonTruthLabelID")
        jet_Wtrk1000_tmp    = getattr(intree,"jet_Wtrk1000")
        jet_Ntrk1000_tmp    = getattr(intree,"jet_Ntrk1000")
        jet_EnergyPerSampling_tmp = getattr(intree,"jet_EnergyPerSampling")
        jet_ConstitE_tmp    = getattr(intree,"jet_ConstitE")
        jet_ChargedFraction_tmp = getattr(intree,"jet_ChargedFraction")
        jet_nMuSeg_tmp      = getattr(intree,"jet_nMuSeg")
        jet_trk_isd_tmp     = getattr(intree,"jet_trk_isd")
        jet_trk_nca_tmp     = getattr(intree,"jet_trk_nca")
        jet_trk_ncasd_tmp   = getattr(intree,"jet_trk_ncasd")
        jet_trk_rg_tmp      = getattr(intree,"jet_trk_rg")
        jet_trk_zg_tmp      = getattr(intree,"jet_trk_zg")
        jet_trk_c1beta02_tmp = getattr(intree,"jet_trk_c1beta02")
        #rhoEMPFLOW_tmp = getattr(intree,"rhoEMPFLOW")
        jet_GhostArea_tmp = getattr(intree,"jet_GhostArea")
        jet_ActiveArea_tmp = getattr(intree,"jet_ActiveArea")
        jet_VoronoiArea_tmp = getattr(intree,"jet_VoronoiArea")
        jet_ActiveArea4vec_pt_tmp = getattr(intree,"jet_ActiveArea4vec_pt")
        jet_ActiveArea4vec_eta_tmp = getattr(intree,"jet_ActiveArea4vec_eta") 
        jet_ActiveArea4vec_phi_tmp = getattr(intree,"jet_ActiveArea4vec_phi") 
        jet_ActiveArea4vec_m_tmp = getattr(intree,"jet_ActiveArea4vec_m") 
        jet_n90Constituents_tmp = getattr(intree,"jet_n90Constituents")
        jet_m_tmp = getattr(intree,"jet_m")           
        jet_true_m_tmp = getattr(intree,"jet_true_m")      
        jet_Ntrk500_tmp = getattr(intree,"jet_Ntrk500")     
        jet_Wtrk500_tmp = getattr(intree,"jet_Wtrk500")    
        jet_Ntrk2000_tmp = getattr(intree,"jet_Ntrk2000")    
        jet_Ntrk3000_tmp = getattr(intree,"jet_Ntrk3000")    
        jet_Ntrk4000_tmp = getattr(intree,"jet_Ntrk4000")    
        jet_Wtrk2000_tmp = getattr(intree,"jet_Wtrk2000")    
        jet_Wtrk3000_tmp = getattr(intree,"jet_Wtrk3000")    
        jet_Wtrk4000_tmp = getattr(intree,"jet_Wtrk4000")    
        jet_ConstitPt_tmp = getattr(intree,"jet_ConstitPt")   
        jet_ConstitEta_tmp = getattr(intree,"jet_ConstitEta")  
        jet_ConstitPhi_tmp = getattr(intree,"jet_ConstitPhi")  
        jet_ConstitMass_tmp = getattr(intree,"jet_ConstitMass") 
        #jet_PileupE_tmp = getattr(intree,"jet_PileupE")     
        #jet_PileupPt_tmp = getattr(intree,"jet_PileupPt")    
        #jet_PileupEta_tmp = getattr(intree,"jet_PileupEta")   
        #jet_PileupPhi_tmp = getattr(intree,"jet_PileupPhi")   
        #jet_PileupMass_tmp = getattr(intree,"jet_PileupMass")  
        #jet_EME_tmp = getattr(intree,"jet_EME")        
        #jet_EMPt_tmp = getattr(intree,"jet_EMPt")        
        #jet_EMEta_tmp = getattr(intree,"jet_EMEta")       
        #jet_EMPhi_tmp = getattr(intree,"jet_EMPhi")       
        #jet_EMMass_tmp = getattr(intree,"jet_EMMass")      
        jet_JESE_tmp = getattr(intree,"jet_JESE")        
        jet_JESPt_tmp = getattr(intree,"jet_JESPt")       
        jet_JESEta_tmp = getattr(intree,"jet_JESEta")      
        jet_JESPhi_tmp = getattr(intree,"jet_JESPhi")      
        jet_JESMass_tmp = getattr(intree,"jet_JESMass")     
        jet_iso_dR_tmp = getattr(intree,"jet_iso_dR")      
        jet_ghostFrac_tmp = getattr(intree,"jet_ghostFrac")
        if njet_tmp != 2:
            continue
        for j in range(0,len(jet_pt_tmp)):
            if np.abs(jet_DetEta_tmp[j]) < np.abs(etaLow) or np.abs(jet_DetEta_tmp[j]) > np.abs(etaHigh):
                continue
            if jet_true_pt_tmp[j] < 200.:
                continue
            runNumber.append(runNumber_tmp)
            eventNumber.append(eventNumber_tmp)
            lumiBlock.append(lumiBlock_tmp)
            bcid.append(bcid_tmp)
            rhoEM.append(rhoEM_tmp)
            mcEventWeight.append(mcEventWeight_tmp)
            NPV.append(NPV_tmp)
            actualInteractionsPerCrossing.append(actualInteractionsPerCrossing_tmp)
            averageInteractionsPerCrossing.append(averageInteractionsPerCrossing_tmp)
            #weight_pileup.append(weight_pileup_tmp)
            weight.append(weight_tmp)
            rho.append(rho_tmp)
            #weight_xs.append(weight_xs_tmp)
            #weight_mcEventWeight.append(weight_mcEventWeight_tmp)
            njet.append(njet_tmp)
            jet_E.append(jet_E_tmp[j])
            jet_pt.append(jet_pt_tmp[j])
            jet_phi.append(jet_phi_tmp[j])
            jet_eta.append(jet_eta_tmp[j])
            jet_DetEta.append(jet_DetEta_tmp[j])
            jet_Jvt.append(jet_Jvt_tmp[j])
            jet_true_pt.append(jet_true_pt_tmp[j])
            jet_true_eta.append(jet_true_eta_tmp[j])
            jet_true_phi.append(jet_true_phi_tmp[j])
            jet_true_e.append(jet_true_e_tmp[j])
            jet_respE.append(jet_respE_tmp[j])
            jet_respPt.append(jet_respPt_tmp[j])
            jet_PartonTruthLabelID.append(jet_PartonTruthLabelID_tmp[j])
            jet_Wtrk1000.append(jet_Wtrk1000_tmp[j])
            jet_Ntrk1000.append(jet_Ntrk1000_tmp[j])
            jet_EnergyPerSampling_0.append(jet_EnergyPerSampling_tmp[j][0])
            jet_EnergyPerSampling_1.append(jet_EnergyPerSampling_tmp[j][1])
            jet_EnergyPerSampling_2.append(jet_EnergyPerSampling_tmp[j][2])
            jet_EnergyPerSampling_3.append(jet_EnergyPerSampling_tmp[j][3])
            jet_EnergyPerSampling_4.append(jet_EnergyPerSampling_tmp[j][4])
            jet_EnergyPerSampling_5.append(jet_EnergyPerSampling_tmp[j][5])
            jet_EnergyPerSampling_6.append(jet_EnergyPerSampling_tmp[j][6])
            jet_EnergyPerSampling_7.append(jet_EnergyPerSampling_tmp[j][7])
            jet_EnergyPerSampling_8.append(jet_EnergyPerSampling_tmp[j][8])
            jet_EnergyPerSampling_9.append(jet_EnergyPerSampling_tmp[j][9])
            jet_EnergyPerSampling_10.append(jet_EnergyPerSampling_tmp[j][10])
            jet_EnergyPerSampling_11.append(jet_EnergyPerSampling_tmp[j][11])
            jet_EnergyPerSampling_12.append(jet_EnergyPerSampling_tmp[j][12])
            jet_EnergyPerSampling_13.append(jet_EnergyPerSampling_tmp[j][13])
            jet_EnergyPerSampling_14.append(jet_EnergyPerSampling_tmp[j][14])
            jet_EnergyPerSampling_15.append(jet_EnergyPerSampling_tmp[j][15])
            jet_EnergyPerSampling_16.append(jet_EnergyPerSampling_tmp[j][16])
            jet_EnergyPerSampling_17.append(jet_EnergyPerSampling_tmp[j][17])
            jet_EnergyPerSampling_18.append(jet_EnergyPerSampling_tmp[j][18])
            jet_EnergyPerSampling_19.append(jet_EnergyPerSampling_tmp[j][19])
            jet_EnergyPerSampling_20.append(jet_EnergyPerSampling_tmp[j][20])
            jet_EnergyPerSampling_21.append(jet_EnergyPerSampling_tmp[j][21])
            jet_EnergyPerSampling_22.append(jet_EnergyPerSampling_tmp[j][22])
            jet_EnergyPerSampling_23.append(jet_EnergyPerSampling_tmp[j][23])
            jet_EnergyPerSampling_24.append(jet_EnergyPerSampling_tmp[j][24])
            jet_EnergyPerSampling_25.append(jet_EnergyPerSampling_tmp[j][25])
            jet_EnergyPerSampling_26.append(jet_EnergyPerSampling_tmp[j][26])
            jet_EnergyPerSampling_27.append(jet_EnergyPerSampling_tmp[j][27])
            jet_ConstitE.append(jet_ConstitE_tmp[j])
            jet_ChargedFraction.append(jet_ChargedFraction_tmp[j])
            jet_nMuSeg.append(jet_nMuSeg_tmp[j])
            jet_trk_isd.append(jet_trk_isd_tmp[j])
            jet_trk_nca.append(jet_trk_nca_tmp[j])
            jet_trk_ncasd.append(jet_trk_ncasd_tmp[j])
            jet_trk_rg.append(jet_trk_rg_tmp[j])
            jet_trk_zg.append(jet_trk_zg_tmp[j])
            jet_trk_c1beta02.append(jet_trk_c1beta02_tmp[j])
            #rhoEMPFLOW.append(rhoEMPFLOW_tmp[j])
            jet_GhostArea.append(jet_GhostArea_tmp[j])
            jet_ActiveArea.append(jet_ActiveArea_tmp[j])
            jet_VoronoiArea.append(jet_VoronoiArea_tmp[j])
            jet_ActiveArea4vec_pt.append(jet_ActiveArea4vec_pt_tmp[j])
            jet_ActiveArea4vec_eta.append(jet_ActiveArea4vec_eta_tmp[j]) 
            jet_ActiveArea4vec_phi.append(jet_ActiveArea4vec_phi_tmp[j]) 
            jet_ActiveArea4vec_m.append(jet_ActiveArea4vec_m_tmp[j]) 
            jet_n90Constituents.append(jet_n90Constituents_tmp[j])
            jet_m.append(jet_m_tmp[j])           
            jet_true_m.append(jet_true_m_tmp[j])
            jet_Ntrk500.append(jet_Ntrk500_tmp[j])
            jet_Wtrk500.append(jet_Wtrk500_tmp[j])
            jet_Ntrk2000.append(jet_Ntrk2000_tmp[j])
            jet_Ntrk3000.append(jet_Ntrk3000_tmp[j])
            jet_Ntrk4000.append(jet_Ntrk4000_tmp[j])
            jet_Wtrk2000.append(jet_Wtrk2000_tmp[j])
            jet_Wtrk3000.append(jet_Wtrk3000_tmp[j])
            jet_Wtrk4000.append(jet_Wtrk4000_tmp[j])
            jet_ConstitPt.append(jet_ConstitPt_tmp[j])
            jet_ConstitEta.append(jet_ConstitEta_tmp[j])
            jet_ConstitPhi.append(jet_ConstitPhi_tmp[j])
            jet_ConstitMass.append(jet_ConstitMass_tmp[j])
            #jet_PileupE.append(jet_PileupE_tmp[j])
            #jet_PileupPt.append(jet_PileupPt_tmp[j])
            #jet_PileupEta.append(jet_PileupEta_tmp[j])
            #jet_PileupPhi.append(jet_PileupPhi_tmp[j])
            #jet_PileupMass.append(jet_PileupMass_tmp[j])
            #jet_EME.append(jet_EME_tmp[j])
            #jet_EMPt.append(jet_EMPt_tmp[j])
            #jet_EMEta.append(jet_EMEta_tmp[j])
            #jet_EMPhi.append(jet_EMPhi_tmp[j])
            #jet_EMMass.append(jet_EMMass_tmp[j])
            jet_JESE.append(jet_JESE_tmp[j])
            jet_JESPt.append(jet_JESPt_tmp[j])
            jet_JESEta.append(jet_JESEta_tmp[j])
            jet_JESPhi.append(jet_JESPhi_tmp[j])
            jet_JESMass.append(jet_JESMass_tmp[j])
            jet_iso_dR.append(jet_iso_dR_tmp[j])
            jet_ghostFrac.append(jet_ghostFrac_tmp[j])
            EventCount.append(branchHist.GetBinContent(1))
            SumWeights.append(branchHist.GetBinContent(3))
            
    outfile = "mc_sample_"+str(sample)
    createh5files(outfile,"runNumber",runNumber,etaLow,etaHigh,jet_type)
    createh5files(outfile,"eventNumber",eventNumber,etaLow,etaHigh,jet_type)
    createh5files(outfile,"lumiBlock",lumiBlock,etaLow,etaHigh,jet_type)
    createh5files(outfile,"bcid",bcid,etaLow,etaHigh,jet_type)
    createh5files(outfile,"rhoEM",rhoEM,etaLow,etaHigh,jet_type)
    createh5files(outfile,"mcEventWeight",mcEventWeight,etaLow,etaHigh,jet_type)
    createh5files(outfile,"NPV",NPV,etaLow,etaHigh,jet_type)
    createh5files(outfile,"actualInteractionsPerCrossing",actualInteractionsPerCrossing,etaLow,etaHigh,jet_type)
    createh5files(outfile,"averageInteractionsPerCrossing",averageInteractionsPerCrossing,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"weight_pileup",weight_pileup,etaLow,etaHigh,jet_type)
    createh5files(outfile,"weight",weight,etaLow,etaHigh,jet_type)
    createh5files(outfile,"rho",rho,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"weight_xs",weight_xs,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"weight_mcEventWeight",weight_mcEventWeight,etaLow,etaHigh,jet_type)
    createh5files(outfile,"njet",njet,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_E",jet_E,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_pt",jet_pt,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_phi",jet_phi,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_eta",jet_eta,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_DetEta",jet_DetEta,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Jvt",jet_Jvt,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_true_pt",jet_true_pt,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_true_eta",jet_true_eta,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_true_phi",jet_true_phi,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_true_e",jet_true_e,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_respE",jet_respE,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_respPt",jet_respPt,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_PartonTruthLabelID",jet_PartonTruthLabelID,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Wtrk1000",jet_Wtrk1000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Ntrk1000",jet_Ntrk1000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_0",jet_EnergyPerSampling_0,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_1",jet_EnergyPerSampling_1,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_2",jet_EnergyPerSampling_2,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_3",jet_EnergyPerSampling_3,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_4",jet_EnergyPerSampling_4,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_5",jet_EnergyPerSampling_5,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_6",jet_EnergyPerSampling_6,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_7",jet_EnergyPerSampling_7,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_8",jet_EnergyPerSampling_8,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_9",jet_EnergyPerSampling_9,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_10",jet_EnergyPerSampling_10,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_11",jet_EnergyPerSampling_11,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_12",jet_EnergyPerSampling_12,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_13",jet_EnergyPerSampling_13,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_14",jet_EnergyPerSampling_14,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_15",jet_EnergyPerSampling_15,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_16",jet_EnergyPerSampling_16,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_17",jet_EnergyPerSampling_17,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_18",jet_EnergyPerSampling_18,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_19",jet_EnergyPerSampling_19,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_20",jet_EnergyPerSampling_20,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_21",jet_EnergyPerSampling_21,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_22",jet_EnergyPerSampling_22,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_23",jet_EnergyPerSampling_23,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_24",jet_EnergyPerSampling_24,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_25",jet_EnergyPerSampling_25,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_26",jet_EnergyPerSampling_26,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_EnergyPerSampling_27",jet_EnergyPerSampling_27,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ConstitE",jet_ConstitE,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ChargedFraction",jet_ChargedFraction,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_nMuSeg",jet_nMuSeg,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_trk_isd",jet_trk_isd,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_trk_nca",jet_trk_nca,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_trk_ncasd",jet_trk_ncasd,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_trk_rg",jet_trk_rg,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_trk_zg",jet_trk_zg,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_trk_c1beta02",jet_trk_c1beta02,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"rhoEMPFLOW",rhoEMPFLOW,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_GhostArea",jet_GhostArea,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ActiveArea",jet_ActiveArea,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_VoronoiArea",jet_VoronoiArea,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ActiveArea4vec_pt",jet_ActiveArea4vec_pt,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ActiveArea4vec_eta",jet_ActiveArea4vec_eta,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ActiveArea4vec_phi",jet_ActiveArea4vec_phi,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ActiveArea4vec_m",jet_ActiveArea4vec_m,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_n90Constituents",jet_n90Constituents,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_m",jet_m,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_true_m",jet_true_m,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Ntrk500",jet_Ntrk500,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Wtrk500",jet_Wtrk500,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Ntrk2000",jet_Ntrk2000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Ntrk3000",jet_Ntrk3000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Ntrk4000",jet_Ntrk4000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Wtrk2000",jet_Wtrk2000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Wtrk3000",jet_Wtrk3000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_Wtrk4000",jet_Wtrk4000,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ConstitPt",jet_ConstitPt,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ConstitEta",jet_ConstitEta,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ConstitPhi",jet_ConstitPhi,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ConstitMass",jet_ConstitMass,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_PileupE",jet_PileupE,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_PileupPt",jet_PileupPt,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_PileupEta",jet_PileupEta,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_PileupPhi",jet_PileupPhi,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_PileupMass",jet_PileupMass,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_EME",jet_EME,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_EMPt",jet_EMPt,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_EMEta",jet_EMEta,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_EMPhi",jet_EMPhi,etaLow,etaHigh,jet_type)
    #createh5files(outfile,"jet_EMMass",jet_EMMass,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_JESE",jet_JESE,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_JESPt",jet_JESPt,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_JESEta",jet_JESEta,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_JESPhi",jet_JESPhi,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_JESMass",jet_JESMass,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_iso_dR",jet_iso_dR,etaLow,etaHigh,jet_type)
    createh5files(outfile,"jet_ghostFrac",jet_ghostFrac,etaLow,etaHigh,jet_type)
    createh5files(outfile,"EventCount",EventCount,etaLow,etaHigh,jet_type)
    createh5files(outfile,"SumWeights",SumWeights,etaLow,etaHigh,jet_type)


parser = OptionParser()
parser.add_option('--inputFileList', default = "fileList.txt", help="List of files which should be included")
parser.add_option('--treeName', default = "IsolatedJet_tree", help="Name of the tree")
parser.add_option('--jettype', default = "LCTopo", type = "string", help = "jet type")
parser.add_option('--etaLow', default = 0, type = "float", help = "lowest eta")
parser.add_option('--etaHigh', default = 0.2, type = "float", help = "highest eta")
parser.add_option('--samples', action="append", type="string")

(opt, args) = parser.parse_args()
if not opt.samples:
    print "Could not find samples, using the default in settings file"
    
for sample in opt.samples:
    print sample
    filename = "make jets"
    if opt.jettype == "LCTopo":
        filename = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/LCTopo/user.jroloff." + str(sample) + ".PythiaMC16dPT00_300920_tree.root/tree_"+ str(sample) + ".root"
    else:
        filename = "/storage/cgrp/atlas_hi/iakova/PhD/QualificationTask/2020_09_30/UFO/user.jroloff." + str(sample) + ".PythiaMC16dPT00_021020_tree.root/tree_"+ str(sample) + ".root"
    createDataset(filename, opt.treeName, sample, opt.etaLow, opt.etaHigh, opt.jettype)


