#include <armadillo>
#include <string>

using std::string;
using arma::rowvec;
using arma::mat;

struct layerParam{
	int level;
	int nodes;
	string activation;
	bool operator<(const layerParam &a){
		return level < a.level;
	}
};

class nnLayerInput{
public:
	nnLayerInput();
	nnLayerInput(int node, rowvec *features,int offset);
	void operator=(const nnLayerInput &);
	bool success(){return _init;};

	void setFeatures(int Nth);
	rowvec out;  // output

	int n_node(){return _node;};
	int n_offset(){return _offset;};
	rowvec* getFeatures(){return _features;};
private:
	bool _init;
	rowvec *_features;

	int _node;
	int _offset;
};

class nnLayerHidden{
public:
	nnLayerHidden();
	nnLayerHidden(
			int node, int input,int wmin, int wmax,
			void (*act)(rowvec &in, rowvec &out, int size),
			void (*dact)(rowvec &in, rowvec &out, int size));
	void operator=(const nnLayerHidden &);
	bool success(){return _init;};
	//forward
	mat weight;
	rowvec sum;  //input*weight
	rowvec out;  //output : o =activation(s)
	//trainig
	rowvec delta;//delta = (higher layer) delta * activation'(s)
	mat wupdate; //wupdate = learning rate * delta * (lower layer) out
	mat wupdates;//wupdate summation : update in a iteration

	void (*act)(rowvec &in, rowvec &out, int size);
	void (*dact)(rowvec &in, rowvec &out, int size);

	int n_node() {return _node;};
	int n_input() {return _input;};

private:
	bool _init;

	int _node;
	int _input;

};

class nnLayerOutput{
public:
	nnLayerOutput();
	nnLayerOutput(
			int node, int input,int wmin, int wmax,
			rowvec *outputs, int offset, int start,
			void (*act)(rowvec &in, rowvec &out, int size),
			void (*dact)(rowvec &in, rowvec &out, int size));

	void operator=(const nnLayerOutput &);
	bool success(){return _init;};

	void setOutput(int Nth);
	rowvec desireOut;//output
	//forward
	mat weight;
	rowvec sum;		//input * weight
	rowvec out;		//output : o =activation(s)
	//trainig
	rowvec delta;	//delta = (higher layer) delta * activation'(s)
	mat wupdate;	//wupdate = lr * delta * (lower layer) out
	mat wupdates;	//wupdate summation : update in a iteration

	//mat error;
	rowvec cost;
	rowvec costnmse;

	void (*act)(rowvec &in, rowvec &out, int size);
	void (*dact)(rowvec &in, rowvec &out, int size);

	int n_node() {return _node;};
	int n_input() {return _input;};

private:
	bool _init;

	int _node;
	int _input;

	int _offset;
	int _start;
	rowvec *_outputs;
};

