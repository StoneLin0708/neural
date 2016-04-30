#pragma once
#define ARMA_NO_DEBUG
#include <vector>
#include <armadillo>
#include <string>
#include "sample.hpp"
#include "algorithm.hpp"

#define dlearning_rate 0.2
#define node_max 100
using std::string;
using std::vector;

namespace nn_t{

	struct activation{
		string name;
		double (*act)(double);
		double (*dact)(double);
	};

	typedef enum{
		empty = 0,
		classification,
		regression,
		timeseries
	}output_t;


	typedef enum {
		all,
		number,
		bunch
	}sampling_t;

	typedef sampling_t testSample_t;


}

struct layerParam{
	int level;
	int nodes;
	string activation;
	bool operator<(const layerParam &a){
		return level < a.level;
	}
};

struct nnParam{
	nn_t::output_t sampleType;			/*n 0*/
	double stopTrainingCost;				/*1*/
	double outputScale;					/*2*/
	string sampleData;					/*n 3*/

	int iteration;						/*n 4*/
	double learningRate;				/*n 5*/
	vector<struct layerParam> hidden;	/*6*/
	struct layerParam output;			/*n 7*/

	string normalizeMethod;				/*8*/
	bool loadWeight;					/*9*/
	bool saveWeight;					/*10*/
	string weightPath;					/*11*/

	string weightName;					/*12*/
	string defaultActivation;			/*n 13*/
	bool testOnly;						/*14*/

	nn_t::sampling_t samplingType;		/*15*/
	int samplingStart;					/*15*/
	int samplingEnd;					/*15*/
	int samplingNumber;					/*15*/

	nn_t::testSample_t testSampleType;	/*16*/
	int testSampleStart;				/*16*/
	int testSampleEnd;					/*16*/
	int testSampleNumber;				/*16*/

	int testStep;						/*17*/
};

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


class nn{
public:
	nn(const string& path);

	bool readSample(const string& path);
	//bool readnn(const string& path);

	void setInput(arma::rowvec &input);
	void setInputSeries(int s);
	void test(); //forward
	void test(int sample); //forward
	void train();

	void testResult();
	void testResultRegression();
	void testResultSeries();

	void error(int &i);
	void show();
	void showd();

	vector<nlayer> layer;

	arma::rowvec outputs;
//	arma::rowvec seriesFeatures;
	vector<arma::rowvec> features;

	int n_sample(){return _n_sample;};

	int n_feature(){return _n_feature;};

	nn_t::output_t type(){return _type;};
	bool load(const string &path);

	nnParam& getParam(){return _param;};
	bool enableParam();

	nn_a::normalizeParam getFNormParam(){return _featureNormParam;};
	nn_a::normalizeParam getONormParam(){return _outputNormParam;};

	vector<double> e;
	bool success(){return _init;};
private:
	nn_a::normalizeParam _featureNormParam;
	nn_a::normalizeParam _outputNormParam;

	int getFirstSample();
	bool getNextSample(int &);
	int getFirstTestSample();
	bool getNextTestSample(int &);
	//bool readLayer(string& in);
	double (*act)(double);
	double (*dact)(double);
	void cal_delm(arma::rowvec &label); //calcualte single data error
	void cal_dels(double label);
	void wupdate(); //update error to weight
	void clear_dels(); //do after update

	int _n_sample;

	int _n_feature;

	bool _init;
	nnParam _param;

	nn_t::output_t _type;

	int iteration;
	double learningRate;

};

