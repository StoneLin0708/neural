#include <armadillo>
#include <string>

using std::string;
using arma::rowvec;
using arma::mat;

namespace nn{

    typedef struct LayerParam{
        int level;
        int nodes;
        string activation;
        bool operator<(const layerParam &a){
            return level < a.level;
        }
    }LayerParam;

    class BaseLayer{
    public:
        BaseLayer();

        int Layer;
        rowvec out;

        const int nodes;

    };

    class CalLayer : public BaseLayer{
	public:
        CalLayer(int node, int input,
				void (*act)(rowvec &in, rowvec &out, int size),
				void (*dact)(rowvec &in, rowvec &out, int size));
        bool success(){return _init;};

		//forward
		mat weight;
		rowvec sum;

		//trainig
		rowvec delta;
		mat wupdate;
		mat wupdates;

        const int n_input() {return input;};
        const int n_node(){return nodes;};

		void act();
		void dact();
        void randomWeight(int wmin, int wmax);

	protected:
		bool _init;

        int input;

		void (*fact)(rowvec &in, rowvec &out, int size);
		void (*fdact)(rowvec &in, rowvec &out, int size);

	};

    class InputLayer : public BaseLayer{
	public:
        InputLayer(int node);

	};

    class HiddenLayer : public CalLayer{
	public:
        HiddenLayer(
                int node, int input,
				void (*act)(rowvec &in, rowvec &out, int size),
				void (*dact)(rowvec &in, rowvec &out, int size));

	};

    class OutputLayer : public CalLayer{
	public:
        OutputLayer(
                int node, int input,
				void (*act)(rowvec &in, rowvec &out, int size),
				void (*dact)(rowvec &in, rowvec &out, int size));


		//mat error;
		rowvec cost;
		rowvec costnmse;


	};

}
