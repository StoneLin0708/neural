#include "core/include/Layer.hpp"
#include <random>
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
        out.zeros(Nodes + 1);
        out(Nodes) = 1;
    }

    void BaseLayer::operator=(const BaseLayer &o){
       Layer = o.Layer;
       Nodes = o.Nodes;
       out = o.out;
    }

    CalLayer::CalLayer(int Layer, int Nodes, int Input, double LearningRate, fun::fact_t act, fun::fact_t dact)
        : BaseLayer( Layer, Nodes){

        Inputs = Input;
        weight.zeros(Input+1, Nodes);
        sum.zeros(Nodes);

        this->LearningRate = LearningRate;
        delta.zeros(Nodes);
        wupdate.zeros(Input+1, Nodes);
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
          << "Weight " << endl << l.weight << endl
          << "Sum " << endl << l.sum << endl
          << "Delta " << endl << l.delta << endl
          << "Wupdate " << endl << l.wupdate << endl
          << "Wupdates " << endl << l.wupdates << endl
          << "Out" << endl << l.out <<endl
          << "-------------------------------------------" <<endl;
        return o;

    }

    void CalLayer::fp(rowvec *In){
        sum = *In * weight;
        out.subvec(0,Nodes-1) = fact(sum , Nodes);
    }

    void CalLayer::update(){
        weight += wupdates;
        wupdates.zeros();
    }

    void CalLayer::RandomWeight(double wmin, double wmax){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(wmin, wmax);

        for(int i=0;i<(int)weight.n_rows;++i)
            for(int j=0;j <(int)weight.n_cols; ++j)
                weight(i,j) = dist(mt);
    }

    InputLayer::InputLayer(int Nodes)
        : BaseLayer(0, Nodes)
    {

    }

    HiddenLayer::HiddenLayer(int Layer, int Nodes, int Input, double LearningRate,
                             fun::fact_t act, fun::fact_t dact)
        : CalLayer(Layer, Nodes, Input, LearningRate, act, dact)
    {

    }

    void HiddenLayer::operator=(const HiddenLayer &o){
        *(static_cast<CalLayer*>(this)) = *(static_cast<const CalLayer*>(&o));
    }

    void HiddenLayer::bp(rowvec *LowOut, CalLayer *UpLayer){
        delta = UpLayer->delta * UpLayer->weight.head_rows(Nodes).t();
        delta %= fdact( sum, Nodes);
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
        out = out.subvec(0,Nodes-1);
    }

    void OutputLayer::operator=(const OutputLayer &o){
        *(static_cast<CalLayer*>(this)) = *(static_cast<const CalLayer*>(&o));
        fcost = o.fcost;
        fdcost = o.fdcost;
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

}
