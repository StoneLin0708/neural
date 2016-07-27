#include "core/include/nn.hpp"
#include "method/include/Method.hpp"
#include "load/include/Loader.hpp"
#include "load/include/SampleFeeder.hpp"

#include <unistd.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cassert>

using namespace std;
using namespace arma;

namespace nn{

Network::Network(){
}

Network::~Network()
{
    assert(Layer.size()>2 || Layer.size() == 0 );
    if(Layer.size() != 0){
        delete NN_GET_INPUT_LAYER(*this);
        for(int i=0; i<NN_GET_HIEEDN_SIZE(*this);++i)
            delete NN_GET_HIEEDN_LAYER(*this, i);
        delete NN_GET_OUTPUT_LAYER(*this);
    }
}

void Network::clear(){
    for(int i=1; i < (int)Layer.size(); ++i){
        static_cast<CalLayer*>(Layer[i])->clear();
    }
}

void Network::fp(){
    for(int i=1; i < (int)Layer.size(); ++i){
        static_cast<CalLayer*>(Layer[i])->fp( &(Layer[i-1]->out) );
    }
}

void Network::bp(){
    for(int i=Layer.size()-1; i>0; --i){
        static_cast<CalLayer*>(Layer[i])->bp( Layer[i-1] );
    }
}

void Network::update(){
    for(int i=Layer.size()-1; i>0; --i){
        static_cast<CalLayer*>(Layer[i])->update();
    }
}

}
