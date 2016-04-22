#include "neural.hpp"
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
	gr.WritePNG("s.png");
	return 0;
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
			n.input.at(0) = x*scale;
			n.input.at(1) = y*scale;
			n.test();
			if(n.oo[0] >= 0.99)
				label = 0;
			else if(n.oo[1] >= 0.99)
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
	gr.WritePNG("r.png");
	return 0;
}

double drawSample3d(sample& sample, string out)
{
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SetRanges(0,10,0,10,0,10);
	gr.Title("sample");
	gr.Rotate(60,50);
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"x",0.5);
	gr.Label('y',"y",0.5);
	gr.Label('z',"z",0.5);

	char flag[10] = " . ";

	mglData	xdat(1), ydat(1), zdat(1);

	for(int i=0; i<(int)sample.size(); i++){
		xdat.a[0] = sample[i].feature[0];
		ydat.a[0] = sample[i].feature[1];
		zdat.a[0] = sample[i].feature[2];
		switch((int)sample[i].l){
			case 0:
				flag[0] = 'r';
				break;
			case 1:
				flag[0] = 'g';
				break;
		}
		gr.Plot(xdat, ydat, zdat, flag);
	}
	gr.WritePNG(out.c_str());
	return 0;
}

double drawResult3d(nn& n,string title,string out)
{
	const int size = 10;
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SetRanges(0,size,0,size,0,size);
	gr.Title(title.c_str());
	gr.Rotate(60,50);
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"x",0.5);
	gr.Label('y',"y",0.5);
	gr.Label('z',"z",0.5);

	char flag[10] = " .";

	mglData	xdat(1), ydat(1), zdat(1);
	int label;
	for(double x=0; x<size; x+=1){
		for(double y=0; y<size; y+=1){
			for(double z=0; z<size; z+=1){
				xdat.a[0] = x;
				ydat.a[0] = y;
				zdat.a[0] = z;
				n.input.at(0) = x/10;
				n.input.at(1) = y/10;
				n.input.at(2) = z/10;
				n.test();
				if(n.oo[0] >= 0.99)
					label = 0;
				else if(n.oo[1] >= 0.99)
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
					gr.Plot(xdat, ydat, zdat, flag);
			}
		}
	}
	gr.WritePNG(out.c_str());
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
	if(argc != 8){
		cout <<" data,iteration,rate,in,hidden,out,data scale" << endl;
		return -1;
	}

	nn n(atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
	n.normalize_scale = atof(argv[7]);
	string path = argv[1];

	if( !n.readSample(path) ){
		cout << "open fail : " << argv[1] << endl;
		return -1;
	}

	n.activation = logistic;
	n.dactivation = dlogistic;
	n.learning_rate = atof( argv[3] );

	n.train( atoi(argv[2]) );

	drawError(n,atoi(argv[2]),"error");

	cv::Mat e = imread("e.png");
	imshow("e",e);

	if(n.getSample()[0].feature.size() > 2){
		waitKey(0);
		return 0;
	}

	string ta = "result i=";
	string tb = argv[2];
	string tc = " lr=";
	string td = argv[3];

	drawSample(n.getSample());
	ta = ta+tb+tc+td;
	drawResult(n,ta,n.normalize_scale);


	cv::Mat m = imread("s.png");
	imshow("s",m);
	cv::Mat r = imread("r.png");
	imshow("r",r);
	waitKey(0);

	return 0;
}
