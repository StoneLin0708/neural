#pragma once
#include <armadillo>
#include <string>
#include <method/include/Method.hpp>
#include <iostream>

using std::string;
using arma::rowvec;
using arma::mat;

namespace nn{

    class BaseLayer{
    public:
        BaseLayer(int Layer, int Nodes);
        void operator=(const BaseLayer &);
        virtual ~BaseLayer(){}

        rowvec out;

        int Layer;
        int Nodes;

    };

    class CalLayer : public BaseLayer{
	public:
        CalLayer(int Layer, int Nodes, int Input, double LearningRate,
                 fun::fact_t act, fun::fact_t dact);
        void operator=(const CalLayer &);
        friend std::ostream& operator<<(std::ostream &, const CalLayer&);
        virtual ~CalLayer(){}

        void RandomWeight(double wmin, double wmax);

        int Inputs;

		//forward
		mat weight;
		rowvec sum;

		//trainig
        double LearningRate;
		rowvec delta;
		mat wupdate;
		mat wupdates;

        virtual void clear();
        void fp(rowvec *In);
        virtual void bp(BaseLayer *LowLayer);
        virtual void update();

        int fpCounter;
        int bpCounter;

	protected:

        fun::fact_t fact;
        fun::fact_t fdact;

	};


    class InputLayer : public BaseLayer{
	public:
        InputLayer(int Nodes);

	};

    class HiddenLayer : public CalLayer{
	public:
        HiddenLayer(int Layer, int Nodes, int Input, double LearningRate,
                 fun::fact_t act, fun::fact_t dact);
        void operator=(const HiddenLayer&);

        void clear();
        void bp(BaseLayer *LowLayer);
        void update();

	};

    class OutputLayer : public CalLayer{
	public:
        OutputLayer(int Layer, int Nodes, int Input, double LearningRate,
                 fun::fact_t act, fun::fact_t dact,
                 fun::fcost_t cost, fun::fcost_t dcost
                    );
        void operator=(const OutputLayer&);

        void clear();
        void bp(BaseLayer *LowLayer);
        void update();

        void CalCost();

        rowvec desire;

        fun::fcost_t fcost;
        fun::fcost_t fdcost;

		rowvec cost;
        rowvec costs;


	};

}
