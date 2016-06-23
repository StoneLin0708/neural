#include "neural.hpp"
#include <iostream>
using namespace std;
/*
void nn::testResultSeries(){
	int s = getFirstTestSample();
	int step = _param.testStep;
	double output, real;
	if(step == 0)
		step = _n_sample;

	int j=0;
	do{
		setInputSeries(s);
		test();
		real =
		( ( features[ s+1 ](0) )
					 /_featureNormParam.scale(0)		)
				+_featureNormParam.average(0);
		output =
		( ( layer.back().o(0))
					/_featureNormParam.scale(0)		)
					+_featureNormParam.average(0);
		if(j<step)
			j++;
		else{
			cout<< " output : " << output<< " real : " << real<< endl;
			cin.get();
			j=0;
		}
	}while(getNextTestSample(s));

}
*/

void nn::testResultClassification(){
	int s;
	int step = _param.testStep;
	int outputClass, realClass;
	int max,i;
	int errors=0;
	sampleSet::param param = {_n_sample, _param.testStart, _param.testEnd, _param.testNumber};
	sampleSet sampleSet(_param.testType, param);

	if(step == 0)
		step = _n_sample;

	int j=0;
	while(!sampleSet.last()){
		s = sampleSet.getNext();
		Linput.setFeatures(s);
		test();
		Loutput.setOutput(s);
		max = 0;
		for(i=0; i<_n_output; ++i){
			if(Loutput.out(i) > Loutput.out(max))
				max = i;
		}
		outputClass = max;
		max = 0;
		for(i=0; i<_n_output; ++i){
			if(Loutput.desireOut(i) > Loutput.desireOut(max))
				max = i;
		}
		realClass = max;
		if(outputClass != realClass)
			errors++;
		if(j<step)
			j++;
		else{
			cout<< " output : " << Loutput.out << ":"<< outputClass << endl;
			cout<< " real : " << Loutput.desireOut <<":"<< realClass<< endl;
			cin.get();
			j=0;
		}
	}

	cout<< " all : "<< _param.testNumber <<" error : " << errors << " predict rate "<< (1-(double)errors/_param.testNumber)*100<<'%' << endl;

}

void nn::testResultRegression(){
	int s;
	int step = _param.testStep;

	sampleSet::param param = {_n_sample, _param.testStart, _param.testEnd, _param.testNumber};
	sampleSet sampleSet(_param.testType, param);

	if(step == 0)
		step = _n_sample;

	int j=0;
	while(!sampleSet.last()){
		s = sampleSet.getNext();
		Linput.setFeatures(s);
		test();
		Loutput.setOutput(s);

		if(j<step)
			j++;
		else{
			cout<< " output : ";
			for(int i=0;i<Loutput.n_node();++i){
				cout << (Loutput.out(i)-outputNormParam.offset)
				/ outputNormParam.scale(i)
				+ outputNormParam.average(i)<<",";
			}
			cout<< " real : ";
			for(int i=0;i<Loutput.n_node();++i){
				cout <<(Loutput.desireOut(i)-outputNormParam.offset)
				/ outputNormParam.scale(i)
				+ outputNormParam.average(i)<<",";
			}
			cin.get();
			j=0;
		}
	}

}

