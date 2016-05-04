#include <mgl2/mgl.h>
#include <math.h>
#include "sample.hpp"
#include "plot.hpp"

using namespace std;

double drawResult(nn& n,string title)
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
	gr.WritePNG((n.getParam().sampleData+"_r.png").c_str());
	system(("display "+ n.getParam().sampleData+ "_r.png").c_str() );
	return 0;
}

double drawError(nn& n, int iteration, string title)
{
	mglGraph gr;
	gr.SetSize(800,600);
	gr.SetRanges(0,iteration,
			0.00001,*max_element(n.e.begin(),n.e.end())*10 );
			//0.1, 100 );
	gr.SetFunc("","lg(y)");
	//gr->Grid("!","h=");
	//gr->FPlot("sqrt(1+x^2)");
	gr.Title(title.c_str());
	gr.Light(true);
	gr.Axis(); gr.Grid(); gr.Box();
	gr.Label('x',"iteration",0.2);
	gr.Label('y',"error",0.2);

	char flag[10] = "b.";
	int s=(int)n.e.size();
	mglData	xdat(s), ydat(s);
	for(int i=0; i<s; ++i){
		xdat.a[i] = i * iteration/n.e.size();
		ydat.a[i] = n.e[i];
	}
	gr.Plot(xdat, ydat,flag);
	gr.WritePNG( (n.getParam().sampleData+"_e.png").c_str());
	system(("display "+ n.getParam().sampleData+ "_e.png").c_str() );
	return 0;
}

