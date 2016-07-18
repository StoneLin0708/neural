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

    CalLayer::CalLayer(int Layer, int Nodes, int Input, double LearningRate, fun::fact_t act, fun::fact_t dact)
        : BaseLayer( Layer, Nodes+1){

        weight.zeros(Input+1, Nodes);
        sum.zeros(Nodes);

        this->LearningRate = LearningRate;
        delta.zeros(Nodes);
        wupdate.zeros(Input+1, Nodes);
        wupdates.zeros(Input+1, Nodes);

        fact = act;
        fdact = dact;
    }

    void CalLayer::fp(rowvec *In){
        sum = *In * weight;
        out = fact(sum , Nodes);
    }

    void CalLayer::update(){
       weight += wupdates;
       wupdates.zeros();
    }

    InputLayer::InputLayer(int Nodes)
        : BaseLayer(0,Nodes)
    {

    }

    HiddenLayer::HiddenLayer(int Layer, int Nodes, int Input, double LearningRate,
                             fun::fact_t act, fun::fact_t dact)
        : CalLayer(Layer, Nodes, Input, LearningRate, act, dact)
    {

    }

    void HiddenLayer::bp(rowvec *LowOut, CalLayer *UpLayer){
       delta = UpLayer->delta * UpLayer->weight.t();
       delta *= fdact( sum, Nodes);
        for(int i=0; i<Nodes; ++i)
            for(int j=0; j<=Inputs; ++j)
                wupdate(j,i) = delta(i) * (*LowOut)(j);
        wupdate *=  LearningRate;
        wupdates -= wupdate;
    }

    OutputLayer::OutputLayer(int Layer, int Nodes, int Input, double LearningRate,
                             fun::fact_t act, fun::fact_t dact,
                             fun::fcost_t cost, fun::fcost_t dcost)
        : CalLayer(Layer, Nodes, Input, LearningRate, act, dact)
    {

        desire.zeros(Nodes);
        this->cost.zeros(Nodes);
        this->fcost = cost;
        this->fdcost = dcost;
    }

    void OutputLayer::bp(rowvec *LowOut){
        cost = fcost(desire, out, Nodes);
        delta = fdcost(desire, out, Nodes) % fdact( sum, Nodes);
        for(int i=0; i<Nodes; ++i)
            for(int j=0; j<=Inputs; ++j)
                wupdate(j,i) = delta(i) * (*LowOut)(j);
        wupdate *=  LearningRate;
        wupdates -= wupdate;
    }


/*CalLayer::CalLayer(CalLayer &&l)
    CalLayer::CalLayer(CalLayer &&l){
        Input = l.Input;

        weight = l.weight;
        sum = l.sum;

        delta = l.delta;
        wupdate = l.wupdate;
        wupdates = l.wupdates;

        fact = l.fact;
        fdact = l.fdact;
    }
*/

/*CalLayer::operator=
    void CalLayer::operator=(const CalLayer &l){
        Input = l.Input;

        weight = l.weight;
        sum = l.sum;

        delta = l.delta;
        wupdate = l.wupdate;
        wupdates = l.wupdates;

        fact = l.fact;
        fdact = l.fdact;

    }
*/

}
