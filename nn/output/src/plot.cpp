#include "output/include/plot.hpp"


#ifdef _WIN32

bool drawResult2D(nn::ANNModel &nnm, bool show){

    return false;
}

#else
#include "layer/include/feedforward.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace nn;

bool drawResult2D(nn::ANNModel &nnm, bool show){

    cv::Mat m(500,500,CV_8UC3,Scalar(255,255,255));

    arma::rowvec in(2);
    for(int y=0;y<500; ++y){
        for(int x=0;x<500; ++x){
            in(0) = x*2/500.0-1;
            in(1) = y*2/500.0-1;
            //nn::Normalize(in, nnm.trainSample.norm_in);
            nnm.network.input() = in;
            nnm.network.fp();
            if(	nnm.network.output()(0) >
                nnm.network.output()(1) ){
                m.at<Vec3b>(y,x) = Vec3b(255,0,0);
            }
            else
                m.at<Vec3b>(y,x) = Vec3b(0,255,0);
        }
    }

    if(show){
        imshow("m",m);
        waitKey(0);
    }

    return true;
}
#endif
/*
#include <mgl2/mgl.h>
#include <math.h>
#include "sample.hpp"
#include "plot.hpp"

using namespace std;

double drawResultTimeseries(nn& n,string title, string name)
{
	mglGraph gr;
	gr.SetSize(1000,600);
	gr.SetRanges(0, n.n_sample(),
			(	(*min_element(
							  &((*n.Linput.getFeatures())(0)),
							  &(*n.Linput.getFeatures())(n.n_sample()+n.getParam().trainFeature-1) ) )
			 - n.featureNormParam.offset )
			/n.featureNormParam.scale(0)
			+n.featureNormParam.average(0)*0.8,
			(	(*max_element(
							  &((*n.Linput.getFeatures())(0)),
							  &(*n.Linput.getFeatures())(n.n_sample()+n.getParam().trainFeature-1) ) )
			 - n.featureNormParam.offset )
			/n.featureNormParam.scale(0)
			+n.featureNormParam.average(0)*1.1
			);
	gr.Title(title.c_str());
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"time",0.2);
	gr.Label('y',"value",0.2);

	char flag[10] = "b.";
	int s = n.n_sample();
	mglData	xdat(s), ydat(s);
	for(int i=0; i<s; ++i){
		xdat.a[i] = i;
		n.Loutput.setOutput(i);
		ydat.a[i] =
			( n.Loutput.desireOut(0) - n.featureNormParam.offset )
			/n.featureNormParam.scale(0)
			+n.featureNormParam.average(0);
	}
	gr.Plot(xdat, ydat,flag);

	flag[0] = 'r';
	for(int i=0; i<s; ++i){
		xdat.a[i] = i;
		n.Linput.setFeatures(i);
		n.test();
		//cout << "d "<< i << " "<<ydat.a[i];
		ydat.a[i] =
			( n.Loutput.out(0) - n.featureNormParam.offset )
			/n.featureNormParam.scale(0)
			+n.featureNormParam.average(0);
		//cout << " t "<<ydat.a[i]<<endl;
	}
	gr.Plot(xdat, ydat,flag);

	gr.WritePNG( (name+"_r.png").c_str());
	//system(("display "+ name + "_r.png").c_str() );
	return 0;
}
double drawResult(nn& n,string title,string name)
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
			n.Linput.out(0) =(
				(x-n.featureNormParam.average(0))*
				n.featureNormParam.scale(0) ) +
				n.featureNormParam.offset;
			n.Linput.out(1) =(
				(y-n.featureNormParam.average(0))*
				n.featureNormParam.scale(0) ) +
				n.featureNormParam.offset;
			n.test();
			if(n.Loutput.out(0) > n.Loutput.out(1) )
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
	flag[1] = '+';
	flag[3] = '4';
	for(int i=0; i<n.n_sample(); ++i){
		n.Linput.setFeatures(i);
		xdat.a[0] =
			( (*n.Linput.getFeatures())(i*2)
			-n.featureNormParam.offset )
			/n.featureNormParam.scale(0)
			+n.featureNormParam.average(0);
		ydat.a[0] =
			( (*n.Linput.getFeatures())(i*2+1)
			-n.featureNormParam.offset )
			/n.featureNormParam.scale(1)
			+n.featureNormParam.average(1);
		n.Loutput.setOutput(i);

		if(n.Loutput.desireOut(0) > n.Loutput.desireOut(1))
			flag[0] = 'R';
		else
			flag[0] = 'G';

		gr.Plot(xdat, ydat,flag);
	}
	gr.WritePNG((name+"_r.png").c_str());
	//system(("display "+ name+ "_r.png").c_str() );
	return 0;
}

double drawError(nn& n, int iteration, string name)
{
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SubPlot(1,2,0);
	gr.Title("mse");
	gr.SetRanges(0,iteration,
			*min_element(n.e.begin(),n.e.end())/10,*max_element(n.e.begin(),n.e.end())*10 );
			//0.1, 100 );
	gr.SetFunc("","lg(y)");
	//gr->Grid("!","h=");
	//gr->FPlot("sqrt(1+x^2)");
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"iteration",0.15);
	gr.Label('y',"error",0.2);

	char flag[10] = "b.";
	int s=(int)n.e.size();
	mglData	xdat(s), ydat(s);
	for(int i=0; i<s; ++i){
		xdat.a[i] = i * iteration/n.e.size();
		ydat.a[i] = n.e[i];
	}
	gr.Plot(xdat, ydat,flag);
//---------------------------------------------------
	gr.SubPlot(1,2,1);
	gr.Title("nmse");
	gr.SetRanges(0,iteration,
			*min_element(n.en.begin(),n.en.end())*0.9,*max_element(n.en.begin(),n.en.end())*1.1 );
	//gr.SetFunc("","lg(y)");
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"iteration",0.15);
	gr.Label('y',"error",0.2);

	flag[0] = 'g';
	for(int i=0; i<s; ++i){
		xdat.a[i] = i * iteration/n.e.size();
		ydat.a[i] = n.en[i];
	}
	gr.Plot(xdat, ydat,flag);

	gr.WritePNG( (name+"_e.png").c_str());
	//system(("display "+ name+ "_e.png").c_str() );
	return 0;
}

*/
