#include "core/include/network.hpp"
#include "method/include/Method.hpp"
#include "load/include/Loader.hpp"
#include "load/include/SampleFeeder.hpp"

#include <iostream>
#include <cassert>

using namespace std;
using namespace arma;

namespace nn{

Network::Network(){
    ivec = nullptr;
    ovec = nullptr;
    dvec = nullptr;
}

Network::~Network()
{
    assert(Layer.size()>2 || Layer.size() == 0 );
    delete ivec;
    delete ovec;
    delete dvec;
    if(Layer.size() != 0){
        while(Layer.size() != 0){
            delete Layer.back();
            Layer.pop_back();
        }
    }
}

void Network::addInputLayer(BaseLayer *l)
{
    Layer.push_back(l);
    ivec = new rowvec(l->out.memptr(),l->Nodes,false,true);
}

void Network::addMiddleLayer(BaseLayer *l){
    Layer.push_back(l);
}

void Network::addOutputLayer(BaseLayer *l, BaseOutputLayer *o){
   Layer.push_back(l);
   OutLayer = o;
   ovec = new rowvec(l->out.memptr(),l->Nodes,false,true);
   dvec = new rowvec(o->desire.memptr(),l->Nodes,false,true);
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
