#include "method/include/Normailze.hpp"
#include <algorithm>
#include <float.h>

using namespace arma;
using namespace std;

namespace nn{

    NormParam Normalize(mat &data, double min, double max){
        NormParam param;
        int size = data.n_cols;
        rowvec fmin(size);
        rowvec fmax(size);

		fmin.fill(DBL_MAX);
		fmax.fill(-DBL_MAX);

        param.average = zeros<rowvec>(size);
        param.scale = zeros<rowvec>(size);
		param.offset = (max + min)/2;

        for(int i=0; i<size; ++i){
            fmin(i) = data.col(i).min();
            fmax(i) = data.col(i).max();

			param.average(i) = ( fmax(i) + fmin(i) )/2;

			if(fmax(i)  == 0 && fmin(i) == 0){
				param.scale(i) = 1;
                param.average(i) = param.offset-DBL_MIN;
			}
			else if(fmax(i) == fmin(i)){
				param.scale(i) = 0;
			}
			else
				param.scale(i) = (max-min)/ ( fmax(i) - fmin(i) );

            data.col(i) -= param.average(i);
            data.col(i) *= param.scale(i);
            data.col(i) += param.offset;
		}
		return param;
    }

    void InvNormalize(mat &data, const NormParam &param){
        for(int i=0; i<(int)data.n_cols;++i){
            data.col(i) -= param.offset;
            data.col(i) /= param.scale(i);
            data.col(i) += param.average(i);
        }
    }

    pair<std::vector<double>, bool> ReMapping(mat &data)
    {
        //if(data.n_cols != 1) return make_pair(vector<double>(), false);
        vector<double> re;
        for(int i=0; i<(int)data.n_rows;++i){
            if( find(re.begin(),re.end(),data(i,0)) == re.end())
                re.push_back(data(i,0));
        }
        mat m(data.n_rows, m.size());
        m.zeros();
        for(int i=0; i<(int)data.n_rows;++i)
            for(int j=0; j<(int)m.size();++j)
                if(re[j]==data(i,0))
                    m(i,j) = 1;
        return make_pair(re,true);
    }

}
