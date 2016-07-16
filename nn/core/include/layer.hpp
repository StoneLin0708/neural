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

	class IOLayer{
	public:
		IOLayer(rowvec *outputs, int offset, int start);

		void set(int Nth);
		rowvec data;


	protected:
		int _offset;
		int _start;
		rowvec *data;

	}

	class CalLayer{
	public:
		CalLayer(int node, int input,int wmin, int wmax,
				void (*act)(rowvec &in, rowvec &out, int size),
				void (*dact)(rowvec &in, rowvec &out, int size));
		bool success(){return _init;};

		//forward
		mat weight;
		rowvec sum;
		rowvec out;

		//trainig
		rowvec delta;
		mat wupdate;
		mat wupdates;

		int n_input() {return _input;};
		int n_node(){return _node;};

		void act();
		void dact();

	protected:
		bool _init;

		int _input;
		int _node;

		void (*fact)(rowvec &in, rowvec &out, int size);
		void (*fdact)(rowvec &in, rowvec &out, int size);

	};

	class LayerInput : public IOLayer{
	public:
		LayerInput(int node, rowvec *features,int offset);

	};

	class LayerHidden : public CalLayer{
	public:
		LayerHidden(
				int node, int input,int wmin, int wmax,
				void (*act)(rowvec &in, rowvec &out, int size),
				void (*dact)(rowvec &in, rowvec &out, int size));

	};

	class LayerOutput : public IOLayer , public CalLayer{
	public:
		LayerOutput(
				rowvec *outputs, int offset, int start,
				int node, int input,int wmin, int wmax,
				void (*act)(rowvec &in, rowvec &out, int size),
				void (*dact)(rowvec &in, rowvec &out, int size));


		//mat error;
		rowvec cost;
		rowvec costnmse;


	};

}
