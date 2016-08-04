#pragma once
#include <core/include/network.hpp>
#include <load/include/SampleFeeder.hpp>

namespace  nn{

class Trainer
{
public:
    Trainer();
    virtual ~Trainer();
    void set(Network *, Sample *);
    void train();

    int iteration;
    double minCost;

    SampleFeeder *sf;

private:
    Network *n;

};

}
