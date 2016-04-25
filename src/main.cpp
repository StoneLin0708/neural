#include "neural.hpp"
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

double logistic(double in){
	return 1/(1+exp(-1*in));
}

double dlogistic(double in){
	double re = logistic(in);
	return re*(1-re);
}

int main(int argc,char* argv[]){
	if(argc != 2){
		cout <<" data" << endl;
		return -1;
	}
	string path = argv[1];
	nn n(path,logistic,dlogistic);
	if(!n.success()) return -2;
	double t0 = omp_get_wtime();
	n.train();
	cout<< " train finish in "
		<< omp_get_wtime() - t0
		<< " sec" << endl;

	drawError(n,n.iteration,"error");

	//return 0;

	cv::Mat e = imread("e.png");
	imshow("e",e);

	if(n.getSample()[0].feature.size() > 2){
		waitKey(0);
		return 0;
	}
	char s[100];
	snprintf(s,100,"result i =%d lr=%f",n.iteration, n.learning_rate);

	drawResult(n,s,n.normalize_scale);

	cv::Mat r = imread("r.png");
	imshow("r",r);
	waitKey(0);

	return 0;
}
