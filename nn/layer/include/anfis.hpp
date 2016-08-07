#pragma once
#include "core/include/Layer.hpp"
#include <vector>
#include <armadillo>
#include <memory>
#include "core/include/network.hpp"

namespace nn {

namespace anfis {

class Membership{
public:
    //Membership();
    double y(double x) const{
        const double xe = x-expect;
        return exp( -1 * ( xe*xe/variance ) );
    }
/*
    double dy(double x) const{
        const double xe = x-expect;
        return -2 * exp( -1 * ( xe*xe/variance ) ) * xe / variance;
    }
*/
    double de(double x) const{
        const double xe = x-expect;
        return 2 * xe / variance * exp( -1 * ( xe*xe/variance ) ) ;
    }

    double dv(double x) const{
        const double xe = x-expect;
        const double sxe = xe * xe;
        return sxe * (1/(variance * variance)) * exp( -1 * ( sxe /variance ) );
    }

    double dele;
    double delv;

    double expect;
    double variance;

    double eupdate;
    double vupdate;
    double eupdates;
    double vupdates;

};

class InputLayer : public BaseLayer{
public:
    InputLayer(int Nodes);
};

class FLayer : public CalLayer{
public:
    FLayer(int Layer, int Input, int MSF, double LR);
    void RandomInit(double,double){}
    void clear();
    void fp(rowvec *in);
    void bp(BaseLayer *LowLayer);
    void update();

    double learningRate;

    int n_msf;

    std::vector<Membership> node;

};

class PLayer : public CalLayer{
public:
    PLayer(int Layer, int Input, int MSF);
    void RandomInit(double,double){}
    void clear(){}
    void fp(rowvec *in);
    void bp(BaseLayer *LowLayer);
    void update(){}

    arma::rowvec fuzzy;

    arma::imat weight;

};

class NLayer : public CalLayer{
public:
    NLayer(int Layer, int Input);
    void RandomInit(double,double){}
    void clear(){}
    void fp(rowvec *in);
    void bp(BaseLayer *LowLayer);
    void update(){}

    double sum;
};

class CLayer : public CalLayer{
public:
    CLayer(int Layer, int Nodes, int Input, arma::rowvec* DataInput, double LR);
    void RandomInit(double,double){}
    void clear();
    void fp(rowvec *in);
    void bp(BaseLayer *LowLayer);
    void update();

    double learningRate;

    arma::rowvec* din;

    arma::mat weight;
    arma::rowvec valf;

    arma::mat wupdate;
    arma::mat wupdates;

};


class OLayer : public CalLayer , public BaseOutputLayer{
public:
    OLayer(int Layer, int Input);
    void CalCost();
    void RandomInit(double,double){}

    void clear();
    void fp(rowvec *in);
    void bp(BaseLayer *LowLayer);
    void update(){}

};

Network* CreateAnfis_Type3(int Input, int MSF, double LR);

}

}
