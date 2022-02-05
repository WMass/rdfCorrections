import uproot
import numpy as np
import numba
import ROOT
import os

def loadXsecsForCharge(variation, charge):
    hists = []
    fracs = fplus if charge == "plus" else fminus
    coeffnames = [f"a{i}" for i in range(8)]
    unpol,binsx,binsy = fracs[f"unpol_{charge}_{variation}"].to_numpy()
    hists.append(np.ones_like(unpol))
    for coeff in coeffnames:
        hist,binsx,binsy = fracs[f"{coeff}_{charge}_{variation}"].to_numpy()
        hists.append(hist)
    coeffs = np.stack(hists, axis=-1)
    xsecs = coeffs*np.expand_dims(unpol, axis=-1)
    return xsecs,binsx,binsy

def loadXsecs(variation, bins=False):
    temp,binsx,binsy = loadXsecsForCharge(variation, "minus")
    xsecs = np.stack((temp, loadXsecsForCharge(variation, "plus")[0]), axis=-2)
    return (xsecs,binsx,binsy) if bins else xsecs

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{script_dir}/../data"
fminus = uproot.open(f"{data_dir}/angularCoefficients/fractions_minus_2022.root")
fplus = uproot.open(f"{data_dir}/angularCoefficients/fractions_plus_2022.root")
nominal,binsy,binspt = loadXsecs("nominal", bins=True)
muRmuFUp = loadXsecs("muRmuFUp")
muRmuFDown = loadXsecs("muRmuFDown")
muRUp = loadXsecs("muRUp")
muRDown = loadXsecs("muRDown")
muFUp = loadXsecs("muFUp")
muFDown = loadXsecs("muFDown")

@numba.jit(nopython=True, nogil=True)
def angularFacs(sintheta, costheta, sinphi, cosphi):
    sin2theta = 2.*sintheta*costheta
    cos2theta = 1-2*sintheta*sintheta
    cos2phi= 1-2*sinphi*sinphi

    return np.array([1.+costheta*costheta,
        0.5*(1.-3.*costheta*costheta), 
        sin2theta*cosphi, 
        0.5*sintheta*sintheta*cos2phi,
        sintheta*cosphi,
        costheta,
        sintheta*sintheta*sin2theta,
        sin2theta*sinphi,
        sintheta*sinphi
    ])*3./16.*np.pi

@numba.jit(nopython=True, nogil=True)
#def makeWeightsFullDim(nominal, variation, sintheta, costheta, sinphi, cosphi):
def makeWeightsFullDim(nominal, variation, sintheta, cost):
    angular = angularFacs(sintheta, costheta, sinphi, cosphi)
    # Annoyingly numba doesn't support a tuple argument for axes
    angexpand = np.expand_dims(np.expand_dims(angular, axis=0), axis=0)
    nom = nominal*angexpand
    var = variation*angexpand

    newshape = (nom.shape[-1], nom.shape[-1])
    # Expand dimension with off diagonal, sum over that to exclude the one index being varied
    nomexp = np.expand_dims(nom, axis=-1)
    offdiag = np.ones(newshape) - np.eye(*newshape)
    offdiagexp = np.expand_dims(np.expand_dims(offdiag, axis=0), axis=0)

    sumnom = np.sum(nomexp*offdiag, axis=-2)
    num = var+sumnom
    denom = np.full(num.shape, np.sum(nom))

    return np.where(denom != 0., num/denom, np.zeros_like(num))

@numba.jit(nopython=True, nogil=True)
# Nominal and variation contain values for 1 pt/y bin
def makeWeights1D(nominal, variation, sintheta, costheta, sinphi, cosphi):
    angular = angularFacs(sintheta, costheta, sinphi, cosphi)
    nom = nominal*angular
    var = variation*angular

    newshape = (nom.shape[-1], nom.shape[-1])
    # Expand dimension with off diagonal, sum over that to exclude the one index being varied
    nomexp = np.expand_dims(nom, axis=-1)
    offdiag = np.ones(newshape) - np.eye(*newshape)

    sumnom = np.sum(nomexp*offdiag, axis=-2)
    num = var+sumnom
    denom = np.full(num.shape, np.sum(nom))

    vals = np.where(denom != 0., num/denom, np.zeros_like(num))
    return vals

@numba.jit(nopython=True, nogil=True)
def findBinIrreg(val, bins):
    bindig = np.digitize(np.array(val, dtype=np.float64), bins)
    binnum = bindig-1
    return fixBin(binnum, bins)

@numba.jit(nopython=True, nogil=True)
def findBinReg(val, bins):
    if bins.shape[-1] < 2:
        return 0
    if val < bins[0]:
        return -1
    step = bins[1]-bins[0]
    binnum = int(np.floor_divide(val, step))
    if binnum >= bins.shape[-1]:
        binnum = bins.shape[-1]-1
    return fixBin(binnum, bins)

@numba.jit(nopython=True, nogil=True)
def fixBin(binnum, bins):
    if (binnum == bins.shape[-1]-1):
        binnum = binnum-1
    elif (binnum < 0):
        binnum = 0
    return binnum

#@numba.jit(nopython=True, nogil=True)
@ROOT.Numba.Declare(["double", "double", "int", "double", "double", "double", "double"], "RVec<double>")
def qcdUncByHelicity(ptW, yW, charge, sintheta, costheta, sinphi, cosphi):
    # Not actually constant, but the last bin is basically overflow so it works
    biny = findBinReg(np.abs(yW), binsy)
    binpt = findBinIrreg(ptW, binspt)
    binq = 1 if charge > 0 else 0
    nom = nominal[biny,binpt,binq]
    return np.concatenate((
        makeWeights1D(nom, muRmuFUp[biny,binpt,binq], sintheta, costheta, sinphi, cosphi),
        makeWeights1D(nom, muRUp[biny,binpt,binq], sintheta, costheta, sinphi, cosphi),
        makeWeights1D(nom, muFUp[biny,binpt,binq], sintheta, costheta, sinphi, cosphi),
        makeWeights1D(nom, muRmuFDown[biny,binpt,binq], sintheta, costheta, sinphi, cosphi),
        makeWeights1D(nom, muRDown[biny,binpt,binq], sintheta, costheta, sinphi, cosphi),
        makeWeights1D(nom, muFDown[biny,binpt,binq], sintheta, costheta, sinphi, cosphi),),
    axis=0)


