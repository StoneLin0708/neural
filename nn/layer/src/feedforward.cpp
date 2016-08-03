#include "layer/include/feedforward.hpp"

namespace nn {
namespace feedforward {

FeedForwardCalLayer::FeedForwardCalLayer(
        int Layer, int Nodes, int Input, double LearningRate,
                         fun::fact_t act, fun::fact_t dact):
    CalLayer(Layer,Nodes,Input){

    weight.zeros(Input+1, Nodes);
    sum.zeros(Nodes);

    this->LearningRate = LearningRate;
    delta.zeros(Nodes);
    wupdate.zeros(Input+1, Nodes);
    wupdates.zeros(Input+1, Nodes);

    fact = act;
    fdact = dact;
}

void FeedForwardCalLayer::RandomInit(double wmin, double wmax){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(wmin, wmax);

    for(int i=0;i<(int)weight.n_rows;++i)
        for(int j=0;j <(int)weight.n_cols; ++j)
            weight(i,j) = dist(mt);
}

void FeedForwardCalLayer::fp(rowvec *In){
    sum = *In * weight;
    out.subvec(0,Nodes-1) = fact(sum , Nodes);
    ++fpCounter;
}

//Input Layer
InputLayer::InputLayer(int Nodes)
    : BaseLayer(0, Nodes){
    out.zeros(Nodes + 1);
    out(Nodes) = 1;
}

//Hidden Layer
HiddenLayer::HiddenLayer(int Layer, int Nodes, int Input, double LearningRate,
                         fun::fact_t act, fun::fact_t dact)
    : FeedForwardCalLayer(Layer, Nodes, Input, LearningRate, act, dact){
    out.zeros(Nodes + 1);
    out(Nodes) = 1;

}

/*
void HiddenLayer::operator=(const HiddenLayer &o){
    *(static_cast<CalLayer*>(this)) = *(static_cast<const CalLayer*>(&o));
}
*/
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
    : FeedForwardCalLayer(Layer, Nodes, Input, LearningRate, act, dact)
    , BaseOutputLayer(Nodes,cost,dcost){
}

/*
void OutputLayer::operator=(const OutputLayer &o){
    *(static_cast<CalLayer*>(this)) = *(static_cast<const CalLayer*>(&o));
    fcost = o.fcost;
    fdcost = o.fdcost;
}
*/

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


}

}
