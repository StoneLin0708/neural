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
    out.zeros(Nodes + 1);
    out(Nodes) = 1;
}

void BaseLayer::operator=(const BaseLayer &o){
   Layer = o.Layer;
   Nodes = o.Nodes;
   out = o.out;
}

//Cal Layer
CalLayer::CalLayer(int Layer, int Nodes, int Input, double LearningRate,
                   fun::fact_t act, fun::fact_t dact)
    : BaseLayer( Layer, Nodes){

    Inputs = Input;
    weight.zeros(Input+1, Nodes);
    sum.zeros(Nodes);

    this->LearningRate = LearningRate;
    delta.zeros(Nodes);
    wupdate.zeros(Input+1, Nodes);
    fpCounter = 0;
    bpCounter = 0;
    wupdates.zeros(Input+1, Nodes);

    fact = act;
    fdact = dact;
}

void CalLayer::operator=(const CalLayer &o){
    *(static_cast<BaseLayer*>(this)) = *(static_cast<const BaseLayer*>(&o));
    weight = o.weight;
    LearningRate = o.LearningRate;
    fact = o.fact;
    fdact = o.fdact;
}

std::ostream& operator<<(std::ostream &o, const CalLayer &l){
    o << "CalLayer "<< l.Layer<< " : In "<< l.Inputs<< " Node "<< l.Nodes<< " LR "<< l.LearningRate <<endl
      << "Weight " << endl << l.weight
      << "Sum " << endl << l.sum
      << "Delta " << endl << l.delta
      << "Wupdate " << endl << l.wupdate
      << "Wupdates " << endl << l.wupdates
      << "Out" << endl << l.out
      << "-------------------------------------------" <<endl;
    return o;

}

void CalLayer::clear(){
    //shout not cross this
    abort();
}

void CalLayer::fp(rowvec *In){
    sum = *In * weight;
    out.subvec(0,Nodes-1) = fact(sum , Nodes);
    ++fpCounter;
}

void CalLayer::bp(BaseLayer *){
    //shout not cross this
    abort();
}

void CalLayer::update(){
    //shout not cross this
    abort();
}

void CalLayer::RandomWeight(double wmin, double wmax){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(wmin, wmax);

    for(int i=0;i<(int)weight.n_rows;++i)
        for(int j=0;j <(int)weight.n_cols; ++j)
            weight(i,j) = dist(mt);
}

//Input Layer
InputLayer::InputLayer(int Nodes)
    : BaseLayer(0, Nodes){
}

//Hidden Layer
HiddenLayer::HiddenLayer(int Layer, int Nodes, int Input, double LearningRate,
                         fun::fact_t act, fun::fact_t dact)
    : CalLayer(Layer, Nodes, Input, LearningRate, act, dact){

}

void HiddenLayer::operator=(const HiddenLayer &o){
    *(static_cast<CalLayer*>(this)) = *(static_cast<const CalLayer*>(&o));
}

void HiddenLayer::clear(){
    wupdates.zeros();
    bpCounter = 0;
    fpCounter = 0;
}

void HiddenLayer::bp(BaseLayer* LowLayer){
    const rowvec &LowOut = LowLayer->out;
    delta %= fdact( sum, Nodes);
    for(int i=0; i<Nodes; ++i)
        for(int j=0; j<=Inputs; ++j)
            wupdate(j,i) = delta(i) * LowOut(j);
    wupdates -= wupdate;
    ++bpCounter;
    if(Layer != 1)
        static_cast<HiddenLayer*>(LowLayer)->delta
                = delta * weight.head_rows(Inputs-1).t();
}

void HiddenLayer::update(){
    weight += wupdates * LearningRate / bpCounter;
}

//Output Layer
OutputLayer::OutputLayer(int Layer, int Nodes, int Input, double LearningRate,
                         fun::fact_t act, fun::fact_t dact,
                         fun::fcost_t cost, fun::fcost_t dcost)
    : CalLayer(Layer, Nodes, Input, LearningRate, act, dact){
    desire.zeros(Nodes);
    this->cost.zeros(Nodes);
    this->costs.zeros(Nodes);
    this->fcost = cost;
    this->fdcost = dcost;
    out = out.subvec(0,Nodes-1);
}

void OutputLayer::operator=(const OutputLayer &o){
    *(static_cast<CalLayer*>(this)) = *(static_cast<const CalLayer*>(&o));
    fcost = o.fcost;
    fdcost = o.fdcost;
}

void OutputLayer::clear(){
    costs.zeros();
    wupdates.zeros();
    bpCounter = 0;
    fpCounter = 0;
}

void OutputLayer::bp(BaseLayer* LowLayer){
    delta = fdcost(desire, out, Nodes) % fdact( sum, Nodes);
    const rowvec &LowOut = LowLayer->out;
    for(int i=0; i<Nodes; ++i)
        for(int j=0; j<=Inputs; ++j)
            wupdate(j,i) = delta(i) * LowOut(j);
    wupdates -= wupdate;
    ++bpCounter;
    static_cast<HiddenLayer*>(LowLayer)->delta = delta * weight.head_rows(Inputs).t();
}

void OutputLayer::update(){
    weight += wupdates * LearningRate / bpCounter;
}

void OutputLayer::CalCost(){
    cost = fcost(desire, out, Nodes);
    costs += cost;
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

}
