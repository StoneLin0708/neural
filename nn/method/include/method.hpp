#pragma once
//#define ARMA_NO_DEBUG
#include <math.h>
#include <armadillo>
#include <string>
#include <tuple>

using arma::rowvec;

namespace nn{

    namespace fun{
        typedef rowvec (*fact_t)(rowvec &in, int size);
        typedef rowvec (*fcost_t)(rowvec &desire, rowvec &out, int size);

        std::tuple<fact_t, fact_t, bool> find_act(std::string &name);
        std::tuple<fcost_t, fcost_t, bool> find_cost(std::string &name);

        rowvec sigmoid(rowvec &in, int size);
        rowvec dsigmoid(rowvec &in, int size);

        rowvec tanh(rowvec &in, int size);
        rowvec dtanh(rowvec &in, int size);

        rowvec mse(rowvec &desire, rowvec &out, int size);
        rowvec dmse(rowvec &desire, rowvec &out, int size);

    }

}
