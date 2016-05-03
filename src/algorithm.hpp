#define ARMA_NO_DEBUG
#include <armadillo>
#include <vector>
using arma::rowvec;
using std::vector;

namespace nn_a{
	typedef struct normalizeParam{
		rowvec average;
		rowvec scale;
		double offset;
	}normParam;

	normParam normalize(rowvec &data, int feature,
			double min, double max);
}
