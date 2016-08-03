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

    double dy(double x) const{
        const double xe = x-expect;
        return -2 * exp( -1 * ( xe*xe/variance ) ) * xe / variance;
    }

    double de(double x) const{
        const double xe = x-expect;
        return 2 * exp( -1 * ( xe*xe/variance ) ) * xe / variance;
    }

    double dv(double x) const{
        const double xe = x-expect;
        const double temp = xe/variance;
        return (exp( -1 * ( xe*xe/variance ) )  * temp * temp);
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

class FPNLayer : public CalLayer{
public:
    FPNLayer(int Layer, int Input, int MSF, double LR);
    void RandomInit(double,double){}
    void clear();
    void fp(rowvec *in);
    void bp(BaseLayer *LowLayer);
    void update();

    double learningRate;

    int n_msf;
    int n_fuzzy;

    std::vector<Membership> node;
    arma::rowvec fuzzy;

    arma::rowvec rule;
    arma::rowvec delta;

    arma::imat weight;

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

    double delta;
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
