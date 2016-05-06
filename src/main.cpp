#include "neural.hpp"
#include "nnio.hpp"
#include "plot.hpp"

#include <string.h>
#include <iostream>
#include <iomanip>
#include <time.h>

#include <omp.h>

using namespace std;
using namespace arma;


int main(int argc,char* argv[]){
	if(argc != 3 && argc != 2){
		cout <<" data  ,result name" << endl;
		return -1;
	}

	string path = argv[1];

	string name;
	if(argc == 3)
		name = argv[2];
	//nn n(path,logistic,dlogistic);
	nnParam param;
	readnn(path,param);
	nn n(param);
	if(!n.success()) return -2;

	double t0 = omp_get_wtime();
	if(!n.gradientChecking())
		return -3;

	n.train();
	//return 0;

	cout<< " train finish in "
		<< omp_get_wtime() - t0
		<< " sec" << endl;

	char s[100];
	snprintf(s,100,"result i =%d lr=%f",n.getParam().iteration, n.getParam().learningRate);
	if(n.type() == nn_t::classification)
		n.testResultClassification();
	if(n.type() == nn_t::timeseries){
		drawResultTimeseries(n,s,name);
	}

	//return 0;
	if(argc == 3)
		drawError(n,n.getParam().iteration,name);
	//onlt draw two feature result
	if(n.type() != nn_t::classification || n.n_feature() != 2)
		return 0;

	drawResult(n,s,name);
	return 0;
}
