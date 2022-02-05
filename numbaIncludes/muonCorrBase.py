import ROOT
import numba
import numpy as np

nparams = 6
netabins = 48

@numba.jit('int64(float64, int64)', nopython=True, nogil=True, debug=True)
def etaBin(eta, nbins):
    etamin = -2.4
    etamax = 2.4
    etastep = (etamax-etamin)/nbins
    if eta < etamin:
        eta = etamin
    elif eta >= etamax:
        eta = etamax-etastep/2
    ieta = int(np.floor((eta-etamin)/etastep))
    return ieta

@ROOT.Numba.Declare(["float", "RVec<float>", "float", "int"], "RVec<double>")
def dummyScaleFrom10MeVMassWeights(eta, massWeights, scale, bins):
    # This is currently hardcoded for W
    upWeight = massWeights[10+1]
    downWeight = massWeights[10-1]
    weightsPerEta = np.ones(bins*2, dtype='float64')
    ieta = etaBin(eta, bins)
    weightsPerEta[ieta] = upWeight if scale == 1. else np.exp(scale*np.log(np.abs(upWeight)*np.sign(upWeight)))
    weightsPerEta[ieta+bins] = downWeight if scale == 1. else np.exp(scale*np.log(np.abs(downWeight))*np.sign(downWeight))
    return weightsPerEta

@numba.jit("float64[:](float64, int64, float64, int64)", nopython=True, nogil=True, debug=True)
def dummyCalibratedPt(pt, ieta, magnitude, etabins):
    ptvars = np.full(etabins*2, pt, dtype='float64') 
    ptvars[ieta] = pt*(1+magnitude)
    ptvars[ieta+etabins] = pt/(1+magnitude)

    return ptvars

@ROOT.Numba.Declare(["float", "float", "float", "int"], "RVec<double>")
def dummyCalibratedPtFlat(pt, eta, magnitude, nbins):
    ieta = etaBin(eta, nbins)
    return dummyCalibratedPt(pt, ieta, magnitude, nbins)

