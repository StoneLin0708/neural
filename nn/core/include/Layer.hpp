#include <armadillo>
#include <string>
#include <method/include/method.hpp>

using std::string;
using arma::rowvec;
using arma::mat;

namespace nn{

    class BaseLayer{
    public:
        BaseLayer(int Layer, int Nodes);

        rowvec out;

        int Layer;
        int Nodes;

    };

    class CalLayer : public BaseLayer{
	public:
        CalLayer(int Layer, int Nodes, int Input,
                 fun::fact_t act, fun::fact_t dact);

        int Input;

		//forward
		mat weight;
		rowvec sum;

		//trainig
		rowvec delta;
		mat wupdate;
		mat wupdates;

		void act();
		void dact();
        void randomWeight(int wmin, int wmax);

        void fp(rowvec *In);
        virtual void bp(rowvec *UpDelta) = 0;

	protected:

        fun::fact_t fact;
        fun::fact_t fdact;

	};

    class InputLayer : public BaseLayer{
	public:
        InputLayer(int node);

	};

    class HiddenLayer : public CalLayer{
	public:
        HiddenLayer(int Layer, int Nodes, int Input,
                 fun::fact_t act, fun::fact_t dact);
        void bp(rowvec *UpDelta);

	};

    class OutputLayer : public CalLayer{
	public:
        OutputLayer(int Layer, int Nodes, int Input,
                 fun::fact_t act, fun::fact_t dact,
                 fun::fcost_t cost, fun::fcost_t dcost
                    );
        void bp(rowvec *UpDelta = nullptr);

        rowvec desire;

        fun::fcost_t fcost;
        fun::fcost_t fdcost;

        //mat error;
		rowvec cost;
		rowvec costnmse;


	};

}
