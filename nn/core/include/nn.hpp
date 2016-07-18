#pragma once
//#define ARMA_NO_DEBUG
#include <vector>
#include <armadillo>
#include <string>
/*
#include "sample.hpp"
#include "algorithm.hpp"
#include "sampleSet.hpp"
*/
#include "core/include/Layer.hpp"

using std::string;
using std::vector;


namespace nn{

    bool gradientChecking(int sample = -1);

    class Network{
    public:
        Network();

        void test(); //forward

        void bp();
        void wupdate();
        void clear_wupdates();

        void error(int &i);

        vector<BaseLayer> Layer;

        /*
        rowvec features;
        rowvec outputs;
        nn_a::normParam featureNormParam;
        nn_a::normParam outputNormParam;
        */

        vector<double> e;
        vector<double> en;
        bool success(){return _init;}

    private:
        bool _init;

        int iteration;

    };

}
