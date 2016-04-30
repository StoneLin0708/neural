#define ARMA_NO_DEBUG
#include <armadillo>
#include <vector>
using arma::rowvec;
using std::vector;

namespace nn_a{
	typedef enum{
		scale
	}normalize_t;

	typedef struct normalizeParam{
		normalize_t type;
		rowvec average;
		rowvec scale;
	}normalizeParam;

	void getNormalizeParam(const vector<rowvec> &s, normalizeParam &param);
	void normalize_o(rowvec &s,const normalizeParam &param);
	void getNormalizeParam_o(const rowvec &s, normalizeParam &param);
	void normalize(vector<rowvec> &s,const normalizeParam &param);
}
