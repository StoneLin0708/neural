#pragma once
#define ARMA_NO_DEBUG
#include <vector>
#include <armadillo>
#include <string>
#include "sample.hpp"

#define dlearning_rate 0.2
#define node_max 100

using std::string;
using std::vector;

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

	arma::mat w;     // weight
	arma::rowvec s;  // input*weight
	arma::rowvec o;  // output : o =activation(s)
	arma::rowvec e;  // error  : desire output - o
	arma::rowvec es; // error summation : error in a iteration
	arma::rowvec d;  // d = (higher layer) d * activation'(s)
	arma::mat del;   // del = learning rate * d * (lower layer) o
	arma::mat dels;  // del summation : del in a iteration

	double (*act)(double in); //activation()
	double (*dact)(double in); //activation'()

	void show();


	void random_w(double min, double max); //random weight

	layer_t type(){return _type;};
	int n_nodes() {return _nodes;};
	int n_input() {return _input;};

	bool success(){return _init;};
private:
	void _initialize(
		layer_t type, int n_input, int n_nodes,
		double (*act)(double in),
		double (*dact)(double in) );

	void _initialize_i(int n_input);

	bool _init;

	layer_t _type;
	int _input;
	int _nodes;
};

struct layerParam{
	nlayer::layer_t type;
	int level;
	int nodes;
	string activation;
};

struct nnParam{
	string sampleData;
	string sampleType;
	string normalizeMethod;
	string defaultActivation;


	bool loadWeight;
	bool saveWeight;
	string loadPath;

	int iteration;
	double learning_rate;
};

class nn{
public:
	typedef enum{
		singleOutput = 0,
		multiOutput = 1
	}label_t;
	//nn(int input_number, int hidden_number, int output_number);
	nn(string& path, double (*activation)(double), double (*dactivation)(double));
	void randomInit();

	bool readSample(const string& path);
	bool readnn(const string& path);

	void setInput(arma::rowvec &input);
	void test(); //forward
	void cal_delm(arma::rowvec &label); //calcualte single data error
	void cal_dels(double label);
	void wupdate(); //update error to weight
	void clear_dels(); //do after update
	void train();

	void error(int i);
	void show();

	double (*act)(double in);
	double (*dact)(double in);

	double learning_rate;
	double normalize_scale;
	double output_scale;
	int iteration;

	vector<nlayer> layer;

	//arma::rowvec de; //desire output : output_num
	//arma::mat label;
	arma::rowvec slabels;
	vector<arma::rowvec> mlabels;
	//arma::mat feature;
	vector<arma::rowvec> features;
	int n_feature;
	int n_label;
	int n_sample;
	label_t type;
	//sample& getSample(){return _s;};

	std::vector<double> e;
	bool success(){return _init;};
private:
	bool _init;
	//sample _s;

	bool readLayer(string& in);
};

