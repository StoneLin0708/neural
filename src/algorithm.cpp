#include "algorithm.hpp"
#include <float.h>

using namespace arma;
using namespace std;
namespace nn_a{
	void getNormalizeParam_o(const rowvec &o, normalizeParam &param){
		int n_feature = o.n_cols;

		rowvec min(1);
		rowvec max(1);

		min.fill(DBL_MAX);
		max.fill(-DBL_MAX);

		param.average = zeros<rowvec>(1);
		param.scale = zeros<rowvec>(1);

		for(int i=0; i<n_feature; ++i){
			if( min(0) > o(i) )
				min(0) = o(i);
			if( max(0) < o(i) )
				max(0) = o(i);
		}
		param.average(0) = ( max(0)+min(0) )/2;
		param.scale(0) = 1/ ( max(0) - min(0) );
		cout<< " max " << max(0)
			<< " min " << min(0)
			<< " ave " << param.average(0)
			<< " sca " << param.scale(0)<<endl;
		cout<<"-----------------"<<endl;
	}

	void normalize_o(rowvec &o,const normalizeParam &param){
		int n_feature = o.n_cols;
		for(int i=0; i<n_feature; ++i){
			//cout << "norm : " << o(i);
			o(i) -= param.average(0);
			o(i) *= param.scale(0);
			o(i) += 0.5;
			//cout << " : " << o(i) << endl;
		}
		normalizeParam tparam;
		getNormalizeParam_o(o,tparam);
	}

	void getNormalizeParam(const vector<rowvec> &s, normalizeParam &param){
		int n_sample = s.size();
		int n_feature = s[0].n_cols;

		rowvec min(n_feature);
		rowvec max(n_feature);

		min.fill(DBL_MAX);
		max.fill(-DBL_MAX);

		param.average = zeros<rowvec>(n_feature);
		param.scale = zeros<rowvec>(n_feature);

		for(int i=0; i<n_feature; ++i){
			for(int j=0; j<n_sample; ++j){
				if( min(i) > s[j](i) )
					min(i) = s[j](i);
				if( max(i) < s[j](i) )
					max(i) = s[j](i);
			}
		}
		param.average = (max + min)/2;
		param.scale = 2 / (max - min);
		cout<< "max" << max
			<< "min" << min
			<< "ave" << param.average
			<< "sca" << param.scale;
		cout<<"-----------------"<<endl;
	}

	void normalize(vector<rowvec> &s,const normalizeParam &param){
		int n_sample = s.size();
		int n_feature = s[0].n_cols;
		for(int i=0; i<n_feature; ++i){
			for(int j=0; j<n_sample; ++j){
				s[j](i) -= param.average(i);
				s[j](i) *= param.scale(i);
			}
		}
		normalizeParam tparam;
		getNormalizeParam(s, tparam);
	}

}
