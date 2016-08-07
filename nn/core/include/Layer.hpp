#pragma once
#include <armadillo>
#include <string>
#include <iostream>
#include "method/include/Method.hpp"

using arma::rowvec;

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

        virtual void RandomInit(double wmin, double wmax) = 0;

        virtual void clear() = 0;
        virtual void fp(rowvec *In) = 0;
        virtual void bp(BaseLayer *LowLayer) = 0;
        virtual void update() = 0;

        rowvec delta;

        int fpCounter;
        int bpCounter;


	};

    class BaseOutputLayer{
    public:
        BaseOutputLayer(int Nodes, fun::fcost_t cost, fun::fcost_t dcost);
        virtual ~BaseOutputLayer(){}

        virtual void CalCost() = 0;

        rowvec desire;
        rowvec cost;
        rowvec costs;

        fun::fcost_t fcost;
        fun::fcost_t fdcost;

    };

}
