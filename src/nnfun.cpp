#include <math.h>
#include "nnfun.hpp"
#include <iostream>
using namespace std;
namespace nn_funa{
	void sigmoid(rowvec &in, rowvec &out, int size){
		for(int i=0; i<size; ++i){
			out(i) = 1/(1+exp(-1*in(i)));
		}
	}

	void dsigmoid(rowvec &in, rowvec &out, int size){
		sigmoid(in, out, size);
		for(int i=0; i<size; ++i){
			out(i) *= (1-out(i));
		}
	}

	void tanh(rowvec &in, rowvec &out, int size){
		for(int i=0; i<size; ++i){
			out(i) = std::tanh( (double)in(i) );
		}

	}
	void dtanh(rowvec &in, rowvec &out, int size){
		nn_funa::tanh(in,out,size);
		for(int i=0; i<size; ++i){
			out(i) = 1 - out(i)*out(i);
		}
	}
/*
	inline
	void softmax(rowvec &in, rowvec &out, int size){
	}

	inline
	void dsoftmax(rowvec &in, rowvec &out, int size){
	}
*/
}

namespace nn_func{
	double mse(double desire, double out){
		return 0.5*pow( (desire-out) , 2);
	}

	double dmse(double desire, double out){
		return out-desire;
	}

	double nmse(double desire, double out){
		return 0.5*pow( (desire-out)/desire , 2);
	}

	double dnmse(double desire, double out){
		return (out-desire)/desire;
	}
}

