#include "../include/AnfisLayer.hpp"
#include <cmath>

using namespace  std;

namespace nn {

namespace anfis {

FuzzyLayer::FuzzyLayer(int Layer, int Nodes, int Input, int MSFs,double LearningRate) :
    CalLayer(Layer, Nodes, Input, LearningRate, nullptr, nullptr){
    wupdate.zeros(Nodes,MSFs*2);
    wupdates.zeros(Nodes,MSFs*2);
    for(int i=0; i<Nodes; ++i){
        node.push_back(vector<Membership>());
        for(int j=0; j<MSFs; ++j){
            node[i].push_back(Membership());
            node[i][j].expect = (j+1) * ((double)1.0/(MSFs+1));
            node[i][j].variance = sqrt((double)1.0/(MSFs+1));
        }
    }
    n_msf = MSFs;
}

void FuzzyLayer::clear(){
    wupdates.zeros();
    fpCounter = 0;
    bpCounter = 0;

}

void FuzzyLayer::fp(rowvec *in){
    for(int i=0; i<Nodes; ++i){
        for(int j=0; j<n_msf; ++j){
            out(i) = node[i][j].y( (*in)(i) );
        }
    }
    ++fpCounter;
}

void FuzzyLayer::bp(BaseLayer *LowLayer){
    const rowvec &LowOut = LowLayer->out;
    for(int i=0; i<Nodes; ++i){
        for(int j=0; j<n_msf; ++j){
            const int col = j*2;
            const Membership &m = node[i][j];
            wupdate(i,col) = m.de( LowOut(i) );
            wupdate(i,col+1) = m.dv( LowOut(i) );
        }
    }
    wupdates += wupdate;
    ++bpCounter;
}

void FuzzyLayer::update(){
    for(int i=0; i<Nodes; ++i){
        for(int j=0; j<n_msf; ++j){
            const int col = j*2;
            Membership &m = node[i][j];
            m.expect = wupdate(i,col);
            m.variance = wupdate(i+1,col);
        }
    }
}

}

}
