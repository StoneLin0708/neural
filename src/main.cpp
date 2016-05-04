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
	if(argc != 2){
		cout <<" data" << endl;
		return -1;
	}
	string path = argv[1];
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
	if(n.type() == nn_t::classification)
		n.testResultClassification();
	/*
	if(n.type() == nn_t::regression)
		n.testResultRegression();
	*/
	//return 0;
	cout<< " train finish in "
		<< omp_get_wtime() - t0
		<< " sec" << endl;

	//return 0;
	drawError(n,n.getParam().iteration,"error");
	//onlt draw two feature result
	//if(n.type() != nn_t::classification || n.n_feature() != 2){

	char s[100];
	snprintf(s,100,"result i =%d lr=%f",n.getParam().iteration, n.getParam().learningRate);

	drawResult(n,s);
	return 0;
}
