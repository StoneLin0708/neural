#pragma once
#include "core/include/nn.hpp"
#include "load/include/Sample.hpp"
#include "load/include/SampleFeeder.hpp"

namespace  nn {

class Tester
{
public:
    Tester();
    ~Tester();
    void set(Network *, Sample *);
    void test();
    bool gradientChecking(bool info=false);

private:
    Network *n;
    SampleFeeder *sf;
};

}
