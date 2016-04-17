#include <vector>
#include <armadillo>
#include <string>

#define input_num 2
#define hidden_num 2
#define output_num 2
#define learning_rate 0.2

//input
//hidden weight
//output

class nn{
public:
	nn();


	bool loadSample(std::string& path);
	void test();
	void train();

	void randomInit();

	void showw();
	void showdw();
	void showd();

	double (*activation)(double in);
	double (*dactivation)(double in);
	double error();

	arma::mat input; //input_num+1
	arma::mat hidden; //input_num+1 hidden_num+1
	arma::mat hs; //input_num
	arma::mat ho; //input_num+1
	arma::mat output; //hidden_num+1 output_num
	arma::mat os; //output_num
	arma::mat oo; //output_num
	arma::mat de; //output_num desired output

	arma::mat odel; //hidden_num+1 output_num
	arma::mat hdel; //input_num+1 hidden_num+1
	//sample;
private:

};

