#include "layer/include/anfis.hpp"
#include <cmath>
#include <random>
#include <iostream>

#define ifnanstop( x , text ){\
        if( (x) != (x) ){\
            std::cout << "nan : " << text << std::endl;\
            cin.get();\
        }\
    }

#define ifoorstop( x , range , text ){\
        if( std::fabs( x ) > range ){\
            std::cout << "oor : " << text << std::endl;\
            cin.get();\
        }\
    }

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

FLayer::FLayer(int Layer, int Inputs, int MSF,double LR)
    : CalLayer(Layer, MSF * Inputs, Inputs){
    learningRate = LR;
    n_msf = MSF;
    for(int i=0; i<Inputs; ++i){
        for(int j=0; j<n_msf; ++j){
            const int n_iter = i*n_msf+j;
            node.push_back(Membership());
            node[n_iter].expect = (j+1) * ((double)1.0/(n_msf+1));
            node[n_iter].variance = (double)1.0/(n_msf+1);
            cout << node[n_iter].expect <<','<< node[n_iter].variance <<endl;
        }
    }
}

void FLayer::clear(){
    for(int i=0;i<Nodes;++i){
        auto &m = node[i];
        m.eupdates=0;
        m.vupdates=0;
    }
    fpCounter=0;
    bpCounter=0;
}

void FLayer::fp(rowvec *in){
    for(int i=0; i<Inputs; ++i){
        for(int j=0; j<n_msf; ++j){
            const int n_iter = i*n_msf+j;
            out(n_iter) = node[n_iter].y( (*in)(i) );
            ifnanstop( out(n_iter) , "F fp out(" + to_string(n_iter)+")" )
            ifoorstop( out(n_iter) , 100 , "F fp max")
        }
    }
    ++fpCounter;
}


void FLayer::bp(BaseLayer *LowLayer){
    const auto &Lout = LowLayer->out;

    //cout << "f bp delta"<<delta <<endl;
    //cin.get();
    for(int i=0; i<Inputs; ++i){
        for(int j=0; j<n_msf; ++j){
            const int n_iter = i*n_msf+j;
            Membership &m = node[n_iter];
            m.dele = m.de( Lout(i) );
            m.delv = m.dv( Lout(i) );

            m.eupdate = delta(n_iter) *  m.dele;
            m.vupdate = delta(n_iter) *  m.delv;

            m.eupdates += m.eupdate;
            m.vupdates += m.vupdate;

            ifnanstop( m.dele , "FPN bp mdele(" + to_string(i+n_msf+j) + ")")
            ifnanstop( m.delv , "FPN bp mdelv(" + to_string(i+n_msf+j) + ")")
        }
    }

    ++bpCounter;
}

void FLayer::update(){
    //cout << "FPN update" <<endl;
    for(int i=0; i<Nodes; ++i){
        Membership &m = node[i];
        //cout << "pm["<<i<<"] e="<<m.expect<<" v="<<m.variance<<" eu="<<m.eupdates<<" vu="<<m.vupdates<<endl;
        if((m.eupdates * learningRate) / bpCounter > 0.1){
            cout << "meu max"<<(m.eupdates * learningRate) / bpCounter<<endl;
            cin.get();
        }
        if((m.vupdates * learningRate) / bpCounter > 0.1){
            cout << "mvu max"<<(m.vupdates * learningRate) / bpCounter<<endl;
            cin.get();
        }

        m.expect -= m.eupdates * learningRate / (double)bpCounter;
        m.variance -= m.vupdates * learningRate / (double)bpCounter;

        ifoorstop( m.expect , 2 , "F update e max")
        ifoorstop( m.variance , 2 , "F update v max")

        //cout << "um["<<i<<"] e="<<m.expect<<" v="<<m.variance<<" eu="<<m.eupdates<<" vu="<<m.vupdates<<endl;
        //cin.get();
    }
}

//----------------------------------------------------------
//

PLayer::PLayer(int Layer, int Input, int MSF)
    :CalLayer(Layer, ipow(MSF, Input/MSF), Input){

    weight.zeros(Nodes, Inputs/MSF);

    for(int i=0;i<Nodes;++i){
        int a=0;
        for(unsigned int j=0;j<weight.n_cols;++j){
            weight(i,j) = a + (i / ipow(MSF,j)) % MSF;
            a+=MSF;
        }
    }
}


void PLayer::fp(rowvec *in){
    for(int i=0; i<Nodes; ++i){
        out(i) = 1;
        //cout<<"rule("<<i<<")=";
        for(unsigned int j=0; j<weight.n_cols; ++j){
            out(i) *= (*in)( weight(i,j) );
            //cout <<"f("<<weight(i,j)<<")*";
        }
        //cout<<"="<<rule(i)<<endl;
        ifnanstop( out(i) , "P fp rule(" + to_string(i)+")" )
        ifoorstop( out(i) , 100 , "P fp rule max")
    }

    ++fpCounter;
}

void PLayer::bp(BaseLayer *LowLayer){
    auto L = static_cast<CalLayer*>(LowLayer);
    L->delta.zeros();
    //cout << "P bp delta" << delta <<endl;
    //cout << "P bp Ldel" << L->delta<<endl;
    for(int i=0; i<Nodes; ++i){
        for(unsigned int j=0;j<weight.n_cols;++j){
            const int &n_iter = weight(i,j);
            L->delta(n_iter) += delta(i) * (out(i) / L->out(n_iter) );
            //cout << "P bp n_iter="<<n_iter<<endl;
        }
    }
    //cout << "P bp Ldel" << L->delta<<endl;

    ++bpCounter;
}

//----------------------------------------------------------
//
NLayer::NLayer(int Layer, int Input)
    :CalLayer(Layer, Input, Input){
}

void NLayer::fp(rowvec *in){
    sum = arma::accu(*in);
    out = *in / sum;
    ++fpCounter;
}

void NLayer::bp(BaseLayer *LowLayer){
    for(int i=0;i<Nodes;++i){
        static_cast<CalLayer*>(LowLayer)->delta(i)
                = delta(i) * ( sum - LowLayer->out(i) ) / (sum * sum) ;
        ifnanstop( delta(i) , "N bp delta(" + to_string(i) + ")")
    }
    ++bpCounter;
}

//----------------------------------------------------------
//
CLayer::CLayer(int Layer, int Nodes, int Input, arma::rowvec *DataInput, double LR) :
    CalLayer(Layer,Nodes,Input){

    weight.zeros(Input+1, Nodes);
    learningRate = LR;
    valf.zeros(Nodes);

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
    //cout << (*din)<<endl;
    //cout << weight <<endl;
    valf = (*din) * weight;
    //cout << valf<<endl;
    //cin.get();
    for(int i=0;i<Nodes;++i)
    ifnanstop( valf(i) , "C fp valf(" + to_string(i) + ")")
    out = valf % (*in);
    for(int i=0;i<Nodes;++i)
    ifnanstop(  (*in)(i) , "C fp in(" + to_string(i) + ")")
    for(int i=0;i<Nodes;++i)
    ifnanstop( out(i) , "C fp out(" + to_string(i) + ")")
    ++fpCounter;
    //cout <<"C   "<< out <<endl;
}

void CLayer::bp(BaseLayer *LowLayer){
    //cout << "C bp" <<endl;
    const auto LowOut = LowLayer->out;
    //cout << *din<<endl;
    for(int i=0; i<Nodes; ++i)
        for(int j=0; j<=Inputs; ++j)
            wupdate(j,i) = delta(i) * (*din)(j) * LowOut(i);

    wupdates += wupdate;

    //cout << "C bp delta"<<delta<<endl;
    static_cast<CalLayer*>(LowLayer)->delta = valf % delta;

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
    auto L = static_cast<CalLayer*>(LowLayer);
    for(int i=0; i<L->Nodes; ++i)
        L->delta(i) = fdcost(desire, out, Nodes)(0);
    ++bpCounter;
}


//----------------------------------------------------------
//
Network* CreateAnfis_Type3(int Input, int MSF, double LR){
    Network* n = new Network;
    n->addInputLayer( new InputLayer(Input)) ;
    n->addMiddleLayer( new FLayer(1, Input, MSF,LR) );
    n->addMiddleLayer( new CLayer(2, n->Layer[1]->Nodes, Input, &(n->Layer[0]->out), LR) );
    auto o = new OLayer(3, Input);
    n->addOutputLayer(static_cast<BaseLayer*>(o),static_cast<BaseOutputLayer*>(o));

    return n;
}


}
}
