#pragma once
#include "core/include/nn.hpp"
#include "load/include/Sample.hpp"
#include "load/include/SampleFeeder.hpp"

namespace  nn {

class Tester
{
public:
    enum TestType{
        non,
        classification,
        regression,
        timeseries
    };

    Tester();
    ~Tester();
    void set(Network *, Sample *);
    void test(TestType type=non);
    void testClassification();
    bool gradientChecking(bool info=false);

private:
    Network *n;
    Sample* s;
    SampleFeeder *sf;
};

}
