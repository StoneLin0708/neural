#include "sample.hpp"
#include "neural.hpp"
#include <iostream>
#include <mgl2/mgl.h>
#include <math.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

double logistic(double in){
	return 1/(1+exp(-1*in));
}

double dlogistic(double in){
	double re = logistic(in);
	return re*(1-re);
}

double drawSample(sample& sample)
{
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SetRanges(0,10,0,10);
	gr.Title("sample");
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"x",0.5);
	gr.Label('y',"y",0.5);

	char flag[10] = " . ";

	mglData	xdat(1), ydat(1);

	for(int i=0; i<(int)sample.size(); i++){
		xdat.a[0] = sample[i].feature[0];
		ydat.a[0] = sample[i].feature[1];
		switch((int)sample[i].l){
			case 0:
				flag[0] = 'r';
				break;
			case 1:
				flag[0] = 'g';
				break;
		}
		gr.Plot(xdat, ydat,flag);
	}
	gr.WriteBMP("s.bmp");
	return 0;
}

double drawResult(nn& n)
{
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SetRanges(0,10,0,10);
	gr.Title("result");
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"x",0.5);
	gr.Label('y',"y",0.5);

	char flag[10] = " . ";

	mglData	xdat(1), ydat(1);
	int label;
	for(double x=0; x<10; x+=0.1){
		for(double y=0; y<10; y+=0.1){
			xdat.a[0] = x;
			ydat.a[0] = y;
			n.input.at(0) = x;
			n.input.at(1) = y;
			n.test();
			if(n.oo[0] >= n.oo[1])
				label = 0;
			else
				label = 1;

			switch(label){
				case 0:
					flag[0] = 'r';
					break;
				case 1:
					flag[0] = 'g';
					break;
			}
			gr.Plot(xdat, ydat,flag);
		}
	}
	gr.WriteBMP("r.bmp");
	return 0;
}

int main(int argc,char* argv[]){
	sample s;
	nn n;

	if( !s.read("data/s1") )
		return -1;

	s.list();

	//n.showw();
	n.activation = logistic;
	n.dactivation = dlogistic;

	for(int j=0; j<1500; ++j){
		if(j%200 == 199){
			drawResult(n);
			Mat r = imread("r.bmp");
			imshow("r",r);
			waitKey(0);
		}
		for(int i=0; i<s.size(); i++){
		//for(int i=0; i<1; i++){
			n.input.at(0) = s[i].feature[0];
			n.input.at(1) = s[i].feature[1];
			n.de.fill(0);
			n.de.at( s[i].l ) = 1;
			n.train();
			/*
			n.showd();
			n.showw();
			n.showdw();
			*/
		}
	}
	n.showw();
	n.showd();
	n.showdw();

	drawSample(s);
	drawResult(n);


	Mat m = imread("s.bmp");
	imshow("s",m);
	Mat r = imread("r.bmp");
	imshow("r",r);
	waitKey(0);

}

