#include "neural.hpp"
#include <string.h>
#include <iostream>
#include <iomanip>
#include <mgl2/mgl.h>
#include <math.h>
#include "sample.hpp"

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

double drawResult(nn& n,string title,double scale)
{
	const int size = 10;
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SetRanges(0,size,0,size);
	gr.Title(title.c_str());
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"x",0.5);
	gr.Label('y',"y",0.5);

	char flag[10] = " . ";

	mglData	xdat(1), ydat(1);
	int label;
	for(double x=0; x<size; x+=0.1){
		for(double y=0; y<size; y+=0.1){
			xdat.a[0] = x;
			ydat.a[0] = y;
			n.layer[0].o(0) = x*scale;
			n.layer[0].o(1) = y*scale;
			n.test();
			if(n.layer.back().o(0) >= 0.99)
				label = 0;
			else if(n.layer.back().o[1] >= 0.99)
				label = 1;
			else
				label = 2;
			switch(label){
				case 0:
					flag[0] = 'r';
					break;
				case 1:
					flag[0] = 'g';
					break;
			}
			if (label != 2)
				gr.Plot(xdat, ydat,flag);
		}
	}
	sample& sample = n.getSample();
	flag[1] = '+';
	flag[3] = '4';
	for(int i=0; i<(int)sample.size(); i++){
		xdat.a[0] = sample[i].feature[0];
		ydat.a[0] = sample[i].feature[1];
		switch((int)sample[i].l){
			case 0:
				flag[0] = 'R';
				break;
			case 1:
				flag[0] = 'G';
				break;
		}
		gr.Plot(xdat, ydat,flag);
	}
	gr.WritePNG("r.png");
	return 0;
}
double drawError(nn& n, int iteration, string title)
{
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SetRanges(0,iteration,
			0,*max_element(n.e.begin(),n.e.end()) );
	gr.Title(title.c_str());
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"iteration",0.5);
	gr.Label('y',"error",0.5);

	char flag[10] = "b. ";

	mglData	xdat(1), ydat(1);
	for(int i=0; i<(int)n.e.size(); ++i){
		xdat.a[0] = i * iteration/n.e.size();
		ydat.a[0] = n.e[i];
		gr.Plot(xdat, ydat,flag);
	}
	gr.WritePNG("e.png");
	return 0;
}

int main(int argc,char* argv[]){
	/*
	for(int i = -20; i<=20; i+=4)
		cout<< i << " : a = " << setw(13) << logistic(i)
			<< " a'= " << setw(13) << dlogistic(i) << endl;
	return 0;
	*/
	if(argc != 2){
		cout <<" data" << endl;
		return -1;
	}
	string path = argv[1];
	nn n(path,logistic,dlogistic);

	n.train();

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
