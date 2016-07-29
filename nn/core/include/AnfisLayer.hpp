#pragma once
#include "core/include/Layer.hpp"
#include <vector>
#include <armadillo>

namespace nn {

namespace anfis {

class Membership{
public:
    //Membership();
    double y(double x) const{
        const double xe = x-expect;
        return exp( -1 * ( xe*xe/variance ) );
    }

    double dy(double x) const{
        const double xe = x-expect;
        return -2 * exp( -1 * ( xe*xe/variance ) ) * xe / variance;
    }

    double de(double x) const{
        const double xe = x-expect;
        return 2 * exp( -1 * ( xe*xe/variance ) ) * xe / variance;
    }

    double dv(double x) const{
        const double xe = x-expect;
        const double temp = xe/variance;
        return (exp( -1 * ( xe*xe/variance ) )  * temp * temp);
    }

    double expect;
    double variance;
};

class FuzzyLayer : public CalLayer
{
public:
    FuzzyLayer(int Layer, int Nodes, int Input, int MSF,double LearningRate);

    void clear();
    void fp(rowvec *in);
    void bp(BaseLayer *LowLayer);
    void update();

    std::vector<std::vector<Membership>> node;
    int n_msf;

};

}

}
