#include <vector>
#include <armadillo>
#include <string>
#include "sample.hpp"

#define dlearning_rate 0.2
#define node_max 50
//input
//hidden weight
//output

class nlayer{
public:
	typedef enum{
		hidden = 1,
		output = 0,
		input = 2
	}layer_t;
	//nlayer();
	nlayer(
		layer_t type, int n_input, int n_nodes = 0,
		double (*activation)(double in) = NULL,
		double (*dactivation)(double in) = NULL);

	arma::mat w; //weight
	arma::rowvec s; //i*w
	arma::rowvec o; //s activation
	arma::rowvec e;
	arma::rowvec es;
	arma::rowvec d;
	arma::mat del;
	arma::mat dels;

	double (*act)(double in);
	double (*dact)(double in);

	void show();
	layer_t type();

	int n_nodes() {return _nodes;};
	int n_input() {return _input;};

	void random_w(double min, double max);

	bool _init;
private:
	void _initialize(
		layer_t type, int n_input, int n_nodes,
		double (*act)(double in),
		double (*dact)(double in) );

	void _initialize_i(int n_input);

	layer_t _type;
	int _input;
	int _nodes;
};

class nn{
public:
	//nn(int input_number, int hidden_number, int output_number);
	nn(std::string& path, double (*activation)(double), double (*dactivation)(double));
	void randomInit();

	bool readSample(std::string& path);
	bool readnn(std::string& path);
	void test(); //forward
	void cal_del(); //calcualte single data error
	void wupdate(); //update error to weight
	void clear_dels(); //do after update
	void train();

	void error(int i);

	double (*act)(double in);
	double (*dact)(double in);

	double learning_rate;

	//matrixs use : size
	std::vector<nlayer> layer;
	arma::rowvec de; //desire output : output_num
	//train matrix
	arma::mat odels; //hidden_num+1 output_num

	sample& getSample();
	std::vector<double> e;
	double normalize_scale;
	int iteration;

private:
	int input_num;
	sample _s;

	bool readLayer(int line, std::string& in);
	void errString(std::string& line, std::string& str,int s ,int e);
	bool readFor(int line, std::string& in, const std::string text, bool test=false);
};

