#pragma once
//#define ARMA_NO_DEBUG
#include <math.h>
#include <armadillo>
using arma::rowvec;

namespace nn_funa{

	void sigmoid(rowvec &in, rowvec &out, int size);

	void dsigmoid(rowvec &in, rowvec &out, int size);

	void tanh(rowvec &in, rowvec &out, int size);

	void dtanh(rowvec &in, rowvec &out, int size);

}

namespace nn_func{
	double mse(double desire, double out);

	double dmse(double desire, double out);

	double nmse(double desire, double out);

	double dnmse(double desire, double out);
}
