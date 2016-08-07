#include "core/include/Layer.hpp"
#include <random>
#include <iostream>
#include <iomanip>
#include <cassert>

using std::cout;
using std::endl;
using arma::randu;
using arma::zeros;

namespace nn{

//Base
BaseLayer::BaseLayer(int Layer, int Nodes){
    this->Layer = Layer;
    this->Nodes = Nodes;
    out.zeros(Nodes);
}

void BaseLayer::operator=(const BaseLayer &o){
   Layer = o.Layer;
   Nodes = o.Nodes;
   out = o.out;
}

//Cal Layer
CalLayer::CalLayer(int Layer, int Nodes, int Input)
    : BaseLayer( Layer, Nodes){

    Inputs = Input;
    fpCounter = 0;
    bpCounter = 0;
    delta.zeros(Nodes);

}
/*
void CalLayer::operator=(const CalLayer &o){
    *(static_cast<BaseLayer*>(this)) = *(static_cast<const BaseLayer*>(&o));
    weight = o.weight;
    LearningRate = o.LearningRate;
    fact = o.fact;
    fdact = o.fdact;
}
*/
std::ostream& operator<<(std::ostream &o, const CalLayer &l){
    o << "CalLayer "<< l.Layer<< " : In "<< l.Inputs<< " Node "<< l.Nodes <<endl
      << "Out" << endl << l.out
      << "-------------------------------------------" <<endl;
    return o;

}

BaseOutputLayer::BaseOutputLayer(int Nodes, fun::fcost_t cost, fun::fcost_t dcost){
    desire.zeros(Nodes);
    this->cost.zeros(Nodes);
    this->costs.zeros(Nodes);
    this->fcost = cost;
    this->fdcost = dcost;
}


}
