import ROOT
import numpy as np
import numba
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{script_dir}/../data/N3LLCorrections"

correctionsWp_file = np.load(f"{data_dir}/inclusive_Wp_pT.npz", allow_pickle=True)
binsWp = correctionsWp_file['bins']
correctionsWp = correctionsWp_file['scetlibCorr3D_Wp']

correctionsWm_file = np.load(f"{data_dir}/inclusive_Wm_pT.npz", allow_pickle=True)
binsWm = correctionsWm_file['bins']
correctionsWm = correctionsWm_file['scetlibCorr3D_Wm']

correctionsZ_file = np.load(f"{data_dir}/inclusive_Z_pT.npz", allow_pickle=True)
binsZ = correctionsZ_file['bins']
correctionsZ = correctionsZ_file['scetlibCorr3D_Z']

# Define all of this outside the function to help Numba with typing
iv,im,iy,ipt = range(4)
# Be careful about typing, numba is nasty about this
binsZ_var = binsZ[iv].astype(np.int64) 
binsZ_m = binsZ[im].astype(np.float64) 
binsZ_y = binsZ[iy].astype(np.float64) 
binsZ_pt = binsZ[ipt].astype(np.float64) 

binsWm_var = binsWm[iv].astype(np.int64)   
binsWm_m = binsWm[im].astype(np.float64)  
binsWm_y = binsWm[iy].astype(np.float64) 
binsWm_pt = binsWm[ipt].astype(np.float64) 

binsWp_var = binsWp[iv].astype(np.int64)   
binsWp_m = binsWp[im].astype(np.float64)  
binsWp_y = binsWp[iy].astype(np.float64) 
binsWp_pt = binsWp[ipt].astype(np.float64) 

@numba.jit('float64[:](float64[:], float64[:], float64[:], boolean, int64)', nopython=True, nogil=True, debug=True)
def correctN3LL_allVars(mV, yV, ptV, norm, chan):
    if chan == 0:
        bins_m = binsZ_m
        bins_y = binsZ_y
        bins_pt = binsZ_pt
        corr = correctionsZ
    elif chan == 1:
        bins_m = binsWm_m
        bins_y = binsWm_y
        bins_pt = binsWm_pt
        corr = correctionsWm
    elif chan == 2:
        bins_m = binsWp_m
        bins_y = binsWp_y
        bins_pt = binsWp_pt
        corr = correctionsWp

    # Numba doesn't seem to have digitize with the first arg a scalar implemented
    binm = np.digitize(mV, bins_m)[0]
    biny = np.digitize(yV, bins_y)[0]
    binpt = np.digitize(ptV, bins_pt)[0]
    if binm == 0 or binm == len(bins_m) or biny == 0 or biny == len(bins_y) or binpt == 0 or binpt == len(bins_pt):
        return np.ones(corr.shape[0])
    varcorr = np.copy(corr[:,binm-1,biny-1,binpt-1])
    # Normalize by central value if you're applying variations to a sample weighted by the central weight
    return varcorr/varcorr[0] if norm else varcorr

@numba.jit('float64(int64, float64[:], float64[:], float64[:], int64)', nopython=True, nogil=True, debug=True)
def correctN3LL(var, mV, yV, ptV, chan):
    if chan == 0:
        bins_m = binsZ_m
        bins_y = binsZ_y
        bins_pt = binsZ_pt
        corr = correctionsZ
    elif chan == 1:
        bins_m = binsWm_m
        bins_y = binsWm_y
        bins_pt = binsWm_pt
        corr = correctionsWm
    elif chan == 2:
        bins_m = binsWp_m
        bins_y = binsWp_y
        bins_pt = binsWp_pt
        corr = correctionsWp

    # Numba doesn't seem to have digitize with the first arg a scalar implemented
    binm = np.digitize(mV, bins_m)[0]
    biny = np.digitize(yV, bins_y)[0]
    binpt = np.digitize(ptV, bins_pt)[0]
    if var >= corr.shape[0] or binm == 0 or binm == len(bins_m) or biny == 0 or biny == len(bins_y) or binpt == 0 or binpt == len(bins_pt):
        return 1.
    return corr[var,binm-1,biny-1,binpt-1]

@ROOT.Numba.Declare(["double", "double", "double", "bool"], "RVec<double>")
def correctN3LL_Z_allVars(mV, yV, ptV, norm=True):
    return correctN3LL_allVars(np.array([mV]), np.array([yV]), np.array([ptV]), norm, 0)

@ROOT.Numba.Declare(["double", "double", "double", "bool"], "RVec<double>")
def correctN3LL_Wm_allVars(mV, yV, ptV, norm=True):
    return correctN3LL_allVars(np.array([mV]), np.array([yV]), np.array([ptV]), norm, 1)

@ROOT.Numba.Declare(["double", "double", "double", "bool"], "RVec<double>")
def correctN3LL_Wp_allVars(mV, yV, ptV, norm=True):
    return correctN3LL_allVars(np.array([mV]), np.array([yV]), np.array([ptV]), norm, 2)

@ROOT.Numba.Declare(["int", "double", "double", "double"], "double")
def correctN3LL_Z(var, mV, yV, ptV):
    return correctN3LL(var, np.array([mV]), np.array([yV]), np.array([ptV]), 0)

@ROOT.Numba.Declare(["int", "double", "double", "double"], "double")
def correctN3LL_Wm(var, mV, yV, ptV):
    return correctN3LL(var, np.array([mV]), np.array([yV]), np.array([ptV]), 1)
#
@ROOT.Numba.Declare(["int", "double", "double", "double"], "double")
def correctN3LL_Wp(var, mV, yV, ptV):
    return correctN3LL(var, np.array([mV]), np.array([yV]), np.array([ptV]), 2)

@ROOT.Numba.Declare(["RVec<int>","RVec<int>","RVec<int>","RVec<int>", "RVec<float>", ], "RVec<bool>")
def prefsrLeptons(status, statusFlags, pdgId, motherIdx, pts):
    pdgIdcopy = pdgId
    leptons = (np.abs(pdgId) >= 11) & (np.abs(pdgIdcopy) <= 16) & (motherIdx >= 0)
    status746 = status == 746
    status23 = status == 23
    motherV = (pdgId[motherIdx] == 23) | (np.abs(pdgIdcopy[motherIdx]) == 24)
    fromHardProcess = ((statusFlags >> 8 ) & 1).astype(np.bool_)

    # Some leptons in MadGraph have no W/Z in the history, but have status 23
    others = leptons & (motherV | status23)
    if np.count_nonzero(others) > 2:
        others = others & fromHardProcess
    
    # If there are status = 746 leptons, they came from photos and are pre-FSR
    photos = leptons & status746
    photos = status746
    nphotos = np.count_nonzero(photos)
    nothers = np.count_nonzero(others)

    if nphotos == 2:
        return photos
    elif nphotos == 1 and nothers == 1:
        return photos | others

    return others


#@ROOT.Numba.Declare(["RVec<int>","RVec<int>","RVec<int>","RVec<int>", "RVec<float>", "RVec<float>", "RVec<float>", "bool"], "RVec<int>")
@numba.jit(nopython=True, nogil=True)
def ewPhotonKinematicsSel(status, statusFlags, pdgId, motherIdx, pts, etas, phis, withISR = False):
    pdgIdcopy = pdgId
    isLepton = (np.abs(pdgId) >= 11) & (np.abs(pdgIdcopy) <= 14) & (motherIdx >= 0)
    isMuon = isLepton & (np.abs(pdgId) == 13)
    isPhoton = pdgId == 22
    status1 = status == 1
    isPrompt = ((statusFlags >> 0 ) & 1).astype(np.bool_)
    isHardProcess = ((statusFlags >> 8 ) & 1).astype(np.bool_)

    leptons = isMuon & status1 & isPrompt & isHardProcess
    photons = isPhoton & status1 & isPrompt
    if not withISR:
        motherV = (pdgId[motherIdx] == 23) | (np.abs(pdgId[motherIdx]) == 24)
        leptons = leptons & motherV
        photons = photons & motherV
    lep = leptons & (pdgId > 0)
    antilep = leptons & (pdgId < 0)

    return (lep + antilep*2 + photons*3).astype(np.int32)
