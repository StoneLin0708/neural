#pragma once
#include <armadillo>
#include <vector>
#include "load/include/Sample.hpp"
#include <map>

using std::vector;
using arma::rowvec;
using arma::mat;

namespace  nn{

    class SampleFeeder{
    public:
        SampleFeeder(Sample *, rowvec *in, rowvec *out);
        virtual ~SampleFeeder();

        virtual void reset();
        virtual void next();
        virtual bool isLast();

        int n_sample;
        int n_input;
        int n_output;

    protected:

        int iter;
        Sample *s;
        rowvec *in;
        rowvec *out;

    };

    typedef SampleFeeder SampleFeeder_Default;

    class SampleFeeder_Classification : public SampleFeeder{
    public:
        SampleFeeder_Classification(Sample *s, rowvec *in, rowvec *out);

        std::map<double,unsigned int> outputMap;

        mat output;
        void reset();
        void next();
        bool isLast();

    };

}
