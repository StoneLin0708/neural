#include <vector>
#include <armadillo>
#include <string>
#include "sample.hpp"

#define input_num 2
#define hidden_num 5
#define output_num 2
#define dlearning_rate 0.2

//input
//hidden weight
//output

class nn{
public:
	nn();
	void randomInit();

	bool readSample(std::string& path);
	void test(); //forward
	void cal_del(); //calcualte single data error
	void wupdate(); //update error to weight
	void clear_dels(); //do after update
	void train(int iteration);


	void showw();
	void showdw();
	void showd();
	void showsd();

	double (*activation)(double in);
	double (*dactivation)(double in);

	double learning_rate;

	//matrixs use : size
	arma::mat input; //input data : input_num+1

	arma::mat hidden; //weight of hidden : input_num+1 hidden_num+1
	arma::mat hs; //hidden out before activation : input_num
	arma::mat ho; //hidden out activated : input_num+1

	arma::mat output; //weight of output : hidden_num+1 output_num
	arma::mat os; //output out before activation : output_num
	arma::mat oo; //output out activated : output_num

	arma::mat de; //desire output : output_num
	//train matrix
	arma::mat od; //output_num
	arma::mat hd; //hidden_num+1
	arma::mat odel; //hidden_num+1 output_num
	arma::mat hdel; //input_num+1 hidden_num+1

	arma::mat ods; //output_num
	arma::mat hds; //hidden_num+1
	arma::mat odels; //hidden_num+1 output_num
	arma::mat hdels; //input_num+1 hidden_num+1

	sample& getSample();
	std::vector<double> e;

private:
	sample _s;

};

