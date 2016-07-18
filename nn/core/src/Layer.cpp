#include "core/include/Layer.hpp"

#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using arma::randu;
using arma::zeros;

namespace nn{

    BaseLayer::BaseLayer(int Layer, int Nodes){
        this->Layer = Layer;
        this->Nodes = Nodes;
        out.zeros(Nodes+1);
        out(Nodes-1) = 1;
    }

    CalLayer::CalLayer(int Layer, int Nodes, int Input, fun::fact_t act, fun::fact_t dact)
        : BaseLayer( Layer, Nodes){
        weight.zeros(Input, Nodes+1);
        sum.zeros(Nodes+1);

        delta.zeros(Nodes+1);
        wupdate.zeros(Input, Nodes+1);
        wupdates.zeros(Input, Nodes+1);

        fact = act;
        fdact = dact;
    }

    void CalLayer::fp(rowvec *In){
        sum = *In * weight;
        out = fact(sum , Nodes);
    }

    void OutputLayer::bp(rowvec *UpDelta){
        fdcost(desire, out, delta, Nodes);
        fdact( sum, out, delta, Nodes);

    }

}
