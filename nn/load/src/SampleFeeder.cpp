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

    SampleFeeder_Classification::SampleFeeder_Classification(Sample *s, rowvec *in, rowvec *out)
        :SampleFeeder(s,in,out){
        assert( n_output == 1);
        if( n_output != 1) abort();

        n_output = out->n_cols;
        output.zeros(s->n_sample, n_output);
        for(int i=0; i<s->n_sample;++i){
            if(ceilf(s->output(i,0)) == s->output(i,0))
                output(i, static_cast<int>(s->output(i,0)) ) = 1;
            else{
                cout << "Classification can't have float output"<<endl;
                abort();}
        }

    }

    void SampleFeeder_Classification::reset(){
        iter = 0;
    }

    void SampleFeeder_Classification::next(){
        for(int i=0; i < n_input; ++i)
           (*in)(i) = s->input(iter, i);
        for(int i=0; i < n_output; ++i)
           (*out)(i) = output(iter, i);

    }

    bool SampleFeeder_Classification::isLast(){
        return iter == n_sample;
    }

}
