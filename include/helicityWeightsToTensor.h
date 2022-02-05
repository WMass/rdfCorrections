#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

// TODO: Template this
//template <size_t T>

const size_t NCOEFFS = 9;
const size_t NVARS = 4; // nominal,muRmuF,muR,muF

//const size_t NPTBINS = 10;
//std::array<double, 11> ptVbins = {0.0, 2.9, 4.7, 6.7, 9.0, 11.8, 15.3, 20.1, 27.2, 40.2, 13000.0};
//size_t ptbin = std::distance(std::begin(ptVbins), std::lower_bound(std::begin(ptVbins), std::end(ptVbins), ptV)) - 1;
Eigen::TensorFixedSize<double, Eigen::Sizes<2, NVARS, 2, NCOEFFS>> helicityWeightsToTensor(ROOT::VecOps::RVec<double> weights, double ptV, int chargebin) {
    Eigen::TensorFixedSize<double, Eigen::Sizes<2, NVARS, 2, NCOEFFS>> out;
    out.setConstant(1.);

    for (size_t i = 0; i < weights.size(); i++) {
        size_t variation = NVARS-1;

        if ((i >= 0 && i < NCOEFFS) || (i >= NCOEFFS*NVARS && i < NCOEFFS*(NVARS+1)))
            variation = 1;
        else if ((i >= NCOEFFS && i < 2*NCOEFFS) || (i >= NCOEFFS*(NVARS+1) && i < NCOEFFS*(NVARS+2)))
            variation = 2;
        // Down = 0, up = 1
        size_t shifttype = i < NCOEFFS*NVARS ? 1 : 0;
        out(chargebin, variation, shifttype, i % NCOEFFS) = weights[i];
    }
    return out;
}
