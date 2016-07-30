#pragma once
#include <armadillo>
#include <string>
#include <iostream>

using std::string;
using arma::rowvec;
using arma::mat;

namespace nn{

    class BaseLayer{
    public:
        BaseLayer(int Layer, int Nodes);
        virtual ~BaseLayer(){}

        void operator=(const BaseLayer &);

        rowvec out;

        int Layer;
        int Nodes;

    };

    class CalLayer : public BaseLayer{
	public:
        CalLayer(int Layer, int Nodes, int Input);
        virtual ~CalLayer(){}

        //void operator=(const CalLayer &);
        friend std::ostream& operator<<(std::ostream &, const CalLayer&);

        int Inputs;

        virtual void RandomInit(double wmin, double wmax);

        virtual void clear() = 0;
        virtual void fp(rowvec *In) = 0;
        virtual void bp(BaseLayer *LowLayer) = 0;
        virtual void update() = 0;

        int fpCounter;
        int bpCounter;


	};


}
