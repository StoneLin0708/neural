#include <math.h>
#include "method/include/method.hpp"

#include <iostream>
#include <cstdlib>

using namespace std;

namespace nn{
    namespace fun{

        static void noact(rowvec &in, rowvec &out, int size){
            abort();
        }

        static void nocost(rowvec &desire, rowvec &out, rowvec &cost, int size){
            abort();
        }

        std::tuple<fact_t, fact_t, bool> find_act(string &name){
            if(name == "sigmoid")
                return make_tuple(sigmoid, dsigmoid, true);
            else if(name == "tanh")
                return make_tuple(tanh, dtanh, true);
            else
                return make_tuple(noact, noact, false);
        }

        std::tuple<fcost_t, fcost_t, bool> find_cost(string &name){
            if(name == "mse")
                return make_tuple(mse, dmse, true);
            else
                return make_tuple(nocost, nocost, false);
        }

        rowvec sigmoid(rowvec &in, int size){
            rowvec out(size);
            for(int i=0; i<size; ++i){
                out(i) = 1/(1+exp(-1*in(i)));
            }
            return out;
        }

        rowvec dsigmoid(rowvec &in, int size){
            rowvec out = sigmoid(in, size);
            for(int i=0; i<size; ++i){
                out(i) *= (1-out(i));
            }
            return out;
        }

        rowvec tanh(rowvec &in, int size){
            rowvec out(size);
            for(int i=0; i<size; ++i){
                out(i) = std::tanh( (double)in(i) );
            }
            return out;
        }

        rowvec dtanh(rowvec &in, int size){
            rowvec out = tanh(in, size);
            for(int i=0; i<size; ++i){
                out(i) = 1 - out(i)*out(i);
            }
            return out;
        }

        rowvec mse(rowvec &desire, rowvec &out, int size){
            rowvec cost(size);
            for(int i=0; i<size; ++i){
                cost(i) = 0.5*pow( (desire(i)-out(i)) , 2);
            }
            return cost;
        }

        rowvec dmse(rowvec &desire, rowvec &out, int size){
            rowvec dcost(size);
            for(int i=0; i<size; ++i){
                dcost(i) = out(i) - desire(i);
            }
            return dcost;
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

	double nmse(double desire, double out){
		return 0.5*pow( (desire-out) , 2)/(desire * out);
	}

	double dnmse(double desire, double out){
		return (out-desire)/(desire*desire);
	}
}


