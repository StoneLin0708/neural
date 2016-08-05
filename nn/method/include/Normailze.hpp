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
        void operator=(const NormParam &p){ average=p.average;scale=p.scale;offset=p.offset;}
    }NormParam;

    NormParam Normalize(mat &data, double min, double max);

    void Normalize(mat &data, const NormParam &param);
    void Normalize(rowvec &data, const NormParam &param);

    void InvNormalize(mat &data, const NormParam &param);
    void InvNormalize(rowvec &data, const NormParam &param);

    std::pair<std::vector<double>, bool> ReMapping(mat &data);
}
