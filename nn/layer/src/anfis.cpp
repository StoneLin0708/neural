#include "layer/include/anfis.hpp"
#include <cmath>
#include <random>

using namespace  std;

namespace nn {

namespace anfis {

InputLayer::InputLayer(int Nodes):BaseLayer(0,Nodes){
    out.zeros(Nodes+1);
    out(Nodes) = 1;
}

//----------------------------------------------------------
//

static int ipow(int x, int y){
    int r=1;
    for(int i=0;i<y;++i)
        r*=x;
    return r;
}

FPNLayer::FPNLayer(int Layer, int Inputs, int MSF,double LR)
    : CalLayer(Layer,ipow(MSF,Inputs),Inputs){
    learningRate = LR;
    n_msf = MSF;
    n_fuzzy = Inputs * MSF;
    fuzzy.zeros(n_fuzzy);
    for(int i=0; i<Inputs; ++i){
        for(int j=0; j<n_msf; ++j){
            node.push_back(Membership());
            node[i*n_msf+j].expect = (j+1) * ((double)1.0/(n_msf+1));
            node[i*n_msf+j].variance = (double)1.0/(n_msf+1);
            cout << node[i*n_msf+j].expect <<','<< node[i*n_msf+j].variance <<endl;
        }
    }

    weight.zeros(Nodes, Inputs);
    rule.zeros(Nodes);

    for(int i=0;i<Nodes;++i){
        int a=0;
        for(int j=0;j<Inputs;++j){
            weight(i,j) = a + i % n_msf;
            a+=n_msf;
        }
    }
    //cout << weight <<endl;
    //cin.get();

}

void FPNLayer::clear(){
    for(int i=0;i<n_fuzzy;++i){
        auto &m = node[i];
        m.eupdates=0;
        m.vupdates=0;
    }
    fpCounter=0;
    bpCounter=0;
}

void FPNLayer::fp(rowvec *in){
    //cout << "FPN fp" <<endl;
    //cout << *in <<endl;
    for(int i=0; i<Inputs; ++i){
        for(int j=0; j<n_msf; ++j){
            fuzzy(i*n_msf+j) = node[i*n_msf+j].y( (*in)(i) );
        }
    }
    //cout << fuzzy<<endl;

    for(int i=0; i<Nodes; ++i){
        rule(i) = 1;
        for(int j=0;j<Inputs;++j){
            rule(i) *= fuzzy( weight(i,j) );
        }
    }
    //cout << rule <<endl;

    double sum = arma::accu(rule);
    out = rule / sum;
    //cout <<"FPN "<< out <<endl;

    ++fpCounter;
}

void FPNLayer::bp(BaseLayer *LowLayer){
    //cout << "FPN bp" <<endl;
    const auto &Lout = LowLayer->out;
    for(int i=0; i<Inputs; ++i){
        for(int j=0; j<n_msf; ++j){
            Membership &m = node[i*n_msf+j];
            m.dele = m.de( Lout(i) );
            m.delv = m.dv( Lout(i) );
        }
    }

        //cout << "d " << delta << " r "<< rule <<endl;
    for(int i=0; i<Nodes; ++i){
        //cout << "d " << delta(i) << " r "<< rule(i) <<endl;
        for(int j=0;j<Inputs;++j){
            const int &node_iter = weight(i,j);
            Membership &m = node[ node_iter ];
            m.eupdate = delta(i) * ( rule(i) / fuzzy(node_iter) )* m.dele;
            m.vupdate = delta(i) * ( rule(i) / fuzzy(node_iter) )* m.delv;
            //cout << "u i "<< i<<",j" << j<<"  "<<m.eupdate << ',' << m.vupdate <<endl;

            m.eupdates += m.eupdate;
            m.vupdates += m.vupdate;
        }
    }

    ++bpCounter;
}

void FPNLayer::update(){
    //cout << "FPN update" <<endl;
    for(int i=0; i<n_fuzzy; ++i){
        Membership &m = node[i];
        //cout << "pm["<<i<<"] e="<<m.expect<<" v="<<m.variance<<" eu="<<m.eupdates<<" vu="<<m.vupdates<<endl;
        m.expect -= (m.eupdates * learningRate) / bpCounter;
        m.variance -= (m.vupdates * learningRate) / bpCounter;
        //cout << "um["<<i<<"] e="<<m.expect<<" v="<<m.variance<<" eu="<<m.eupdates<<" vu="<<m.vupdates<<endl;
        cin.get();
    }
}

//----------------------------------------------------------
//
CLayer::CLayer(int Layer, int Nodes, int Input, arma::rowvec *DataInput, double LR) :
    CalLayer(Layer,Nodes,Input){

    weight.zeros(Input+1, Nodes);
    learningRate = LR;
    valf.zeros(Nodes);

    delta = 0;
    wupdate.zeros(Input+1, Nodes);
    wupdates.zeros(Input+1, Nodes);
    din = DataInput;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1, 1);

    for(int i=0;i<(int)weight.n_rows;++i)
        for(int j=0;j <(int)weight.n_cols; ++j)
            weight(i,j) = dist(mt);

}

void CLayer::clear(){
    wupdates.zeros();
    fpCounter=0;
    bpCounter=0;
}

void CLayer::fp(rowvec *in){
    //cout << "C fp" <<endl;
    valf = (*din) * weight;
    out = valf % (*in);
    ++fpCounter;
    //cout <<"C   "<< out <<endl;
}

void CLayer::bp(BaseLayer *LowLayer){
    //cout << "C bp" <<endl;
    const auto LowOut = LowLayer->out;
    //cout << *din<<endl;
    for(int i=0; i<Nodes; ++i)
        for(int j=0; j<=Inputs; ++j)
            wupdate(j,i) = (*din)(j) * LowOut(i);

    wupdates += wupdate * delta;

    static_cast<FPNLayer*>(LowLayer)->delta = delta * valf;

    ++bpCounter;
}

void CLayer::update(){
    //cout << " C update" <<endl;
    weight -= wupdates * learningRate;
    //cout << wupdates <<endl;

}


//----------------------------------------------------------
//
OLayer::OLayer(int Layer, int Inputs):
    CalLayer(Layer, 1, Inputs),
    BaseOutputLayer(1, fun::mse, fun::dmse){

}

void OLayer::CalCost(){
    cost = fcost(desire,out,1);
    costs += cost;
}

void OLayer::clear(){
    costs.zeros();
    fpCounter=0;
    bpCounter=0;
}

void OLayer::fp(rowvec *in){
    out(0) = accu(*in);
    //cout <<desire<<endl;
    //cout <<"O   "<< out <<endl;
    ++fpCounter;
}

void OLayer::bp(BaseLayer *LowLayer){
    static_cast<CLayer*>(LowLayer)->delta = fdcost(desire,out,1)(0);
    ++bpCounter;
}


//----------------------------------------------------------
//
Network* CreateAnfis_Type3(int Input, int MSF, double LR){
    Network* n = new Network;
    n->addInputLayer( new InputLayer(Input)) ;
    n->addMiddleLayer( new FPNLayer(1, Input, MSF,LR) );
    n->addMiddleLayer( new CLayer(2, n->Layer[1]->Nodes, Input, &(n->Layer[0]->out), LR) );
    auto o = new OLayer(3, Input);
    n->addOutputLayer(static_cast<BaseLayer*>(o),static_cast<BaseOutputLayer*>(o));

    return n;
}


}
}
