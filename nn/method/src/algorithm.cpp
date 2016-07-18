#include "method/include/algorithm.hpp"
#include <float.h>

using namespace arma;
using namespace std;
namespace nn_a{

	normParam normalize(rowvec &data, int feature,
			double min, double max){
		normParam param;
		rowvec fmin(feature);
		rowvec fmax(feature);

		fmin.fill(DBL_MAX);
		fmax.fill(-DBL_MAX);

		param.average = zeros<rowvec>(feature);
		param.scale = zeros<rowvec>(feature);

		int sample = data.n_cols/feature;
		param.offset = (max + min)/2;
		for(int i=0; i<feature; ++i){
			for(int j=0; j<sample; ++j){
				if( fmin(i) > data(j*feature+i) )
					fmin(i) = data(j*feature+i);
				if( fmax(i) < data(j*feature+i) )
					fmax(i) = data(j*feature+i);
			}
			param.average(i) = ( fmax(i) + fmin(i) )/2;
			if(fmax(i)  == 0 && fmin(i) == 0){
				param.scale(i) = 1;
				param.average(i) = param.offset-0.0000000001;
			}
			else if(fmax(i) == fmin(i)){
				param.scale(i) = 0;
			}
			else
				param.scale(i) = (max-min)/ ( fmax(i) - fmin(i) );

			for(int j=0; j<sample; ++j){
				data(j*feature+i) -= param.average(i);
				data(j*feature+i) *= param.scale(i);
				data(j*feature+i) += param.offset;
			}
		}
		return param;
		//test
		cout<< " max " << fmax
			<< " min " << fmin
			<< " ave " << param.average
			<< " sca " << param.scale
			<< " off " << param.offset << endl;
		cout<<"-----------------"<<endl;
		rowvec taverage = zeros<rowvec>(feature);
		rowvec tscale = zeros<rowvec>(feature);
		fmin.fill(DBL_MAX);
		fmax.fill(-DBL_MAX);

		for(int i=0; i<feature; ++i){
			for(int j=0; j<sample; ++j){
				if( fmin(i) > data(j*feature+i) )
					fmin(i) = data(j*feature+i);
				if( fmax(i) < data(j*feature+i) )
					fmax(i) = data(j*feature+i);
			}
			taverage(i) = ( fmax(i) + fmin(i) )/2;
			tscale(i) = (max-min)/ ( fmax(i) - fmin(i) );
		}

		cout<< " max " << fmax
			<< " min " << fmin
			<< " ave " << taverage
			<< " sca " << tscale <<endl;
		cout<<"-----------------"<<endl;
		return param;
	}

}
