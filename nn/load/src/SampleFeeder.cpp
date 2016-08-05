#include "load/include/SampleFeeder.hpp"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <math.h>

using std::endl;
using std::cout;

namespace nn{

    SampleFeeder::SampleFeeder(Sample *s, rowvec *in, rowvec *out){
        this->s = s;
        this->in = in;
        this->out = out;
        n_sample = s->n_sample;
        n_input = s->n_input;
        n_output = s->n_output;
        reset();
    }

    SampleFeeder::~SampleFeeder(){
    }

    void SampleFeeder::reset(){
        iter = 0;
    }

    void SampleFeeder::next(){
        for(int i=0; i < n_input; ++i)
           (*in)(i) = s->input(iter, i);
        for(int i=0; i < n_output; ++i)
           (*out)(i) = s->output(iter, i);
        ++iter;
    }

    bool SampleFeeder::isLast(){
        return iter == n_sample;
    }

}
