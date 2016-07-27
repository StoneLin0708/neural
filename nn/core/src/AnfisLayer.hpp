#pragma once
#include "core/include/Layer.hpp"
#include <vector>
namespace nn {

namespace anfis {

class Membership{
public:
    Membership();
    double y(double x){
        return exp( -1 * ( powf(x-expect, 2) / (2*powf(variance, 2)) ) );
    }
    double expect;
    double variance;
};

class FS2M{
public:
    Membership m[2];
};

class FuzzyLayer : public BaseLayer
{
public:
    FuzzyLayer();
    void fp();
    std::vector<FS2M> node;

};

}

}
