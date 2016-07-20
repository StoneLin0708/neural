#pragma once
#include <armadillo>
#include <string>
#include <method/include/method.hpp>
#include <iostream>

using std::string;
using arma::rowvec;
using arma::mat;

namespace nn{

    class BaseLayer{
    public:
        BaseLayer(int Layer, int Nodes);
        void operator=(const BaseLayer &);

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


        int Inputs;

		//forward
		mat weight;
		rowvec sum;

		//trainig
        double LearningRate;
		rowvec delta;
		mat wupdate;
		mat wupdates;

        void update();
		void act();
		void dact();
        void RandomWeight(double wmin, double wmax);

        void fp(rowvec *In);

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
        void bp(rowvec *LowOut, CalLayer *UpLayer);

	};

    class OutputLayer : public CalLayer{
	public:
        OutputLayer(int Layer, int Nodes, int Input, double LearningRate,
                 fun::fact_t act, fun::fact_t dact,
                 fun::fcost_t cost, fun::fcost_t dcost
                    );
        void operator=(const OutputLayer&);
        void bp(rowvec *LowOut);

        rowvec desire;

        fun::fcost_t fcost;
        fun::fcost_t fdcost;

		rowvec cost;


	};

}
