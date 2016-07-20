#pragma once
#include <armadillo>
#include <vector>
#include <tuple>

using arma::mat;
using arma::rowvec;
using std::vector;

namespace nn{
    typedef struct NormParam{
		rowvec average;
		rowvec scale;
		double offset;
    }NormParam;

    NormParam Normalize(mat &data, double min, double max);
    void InvNormalize(mat &data, const NormParam &param);
    std::pair<std::vector<double>, bool> ReMapping(mat &data);
}
