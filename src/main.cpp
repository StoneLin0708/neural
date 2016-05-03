#include "neural.hpp"
#include "nnio.hpp"
#include "plot.hpp"
//#include "sample.hpp"

#include <string.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
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
	n.testResultClassification();
	/*
	if(n.type() == nn_t::classification)
		n.testResult();
	if(n.type() == nn_t::regression)
		n.testResultRegression();
		*/
	//n.showd();
	//return 0;
	cout<< " train finish in "
		<< omp_get_wtime() - t0
		<< " sec" << endl;

	return 0;
	drawError(n,n.getParam().iteration,"error");
	cv::Mat e = imread(
		(n.getParam().sampleData+"_e.png").c_str()
		);
	imshow("e",e);
	//onlt draw two feature result
	//if(n.type() != nn_t::classification || n.n_feature() != 2){

	if(true){
		waitKey(0);
		return 0;
	}

	char s[100];
	snprintf(s,100,"result i =%d lr=%f",n.getParam().iteration, n.getParam().learningRate);

	drawResult(n,s);
	cv::Mat r = imread(
		(n.getParam().sampleData+"_r.png").c_str()
	);
	imshow("r",r);

	waitKey(0);
	return 0;
}
