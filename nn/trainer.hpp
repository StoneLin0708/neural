#pragma once
#include <core/include/nn.hpp>
#include <load/include/SampleFeeder.hpp>

namespace  nn{

    class Trainer
    {
    public:
        Trainer();
        virtual ~Trainer();
        void set(Network *, Sample *);
        void train();
        Network *n;
        SampleFeeder *sf;

        int iteration;
        double minCost;
    };

}
