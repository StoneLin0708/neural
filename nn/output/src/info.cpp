/*
#include "neural.hpp"
#include <iostream>
#include <iomanip>
#include <float.h>
#include <sstream>
using namespace std;
void nn::show(){
	cout << "----------" << "-input--" << "----------" << endl;
	cout<< " input feature +1 bias: "<< Linput.n_node() << endl;

	for(int i=0;i<(int)Lhidden.size();++i){
	cout << "----------" << "hidden"<<setw(2)<< i+1 << "----------" << endl;
	cout<< " input : "<< Lhidden[i].weight.n_rows
		<< " nodes : "<< Lhidden[i].weight.n_cols
		<< " activation : "<< _param.hidden[i].activation << endl;
	}
	cout << "----------" << "-output-" << "----------" << endl;
	cout<< " input : "<< Loutput.weight.n_rows
		<< " nodes : "<< Loutput.weight.n_cols << endl;

	cout << "----------" << "--------" << "----------" << endl;
}
void nn::showParam(){
	string stopTrainingCost;
	stringstream ss;
	if(_param.stopTrainingCost == -DBL_MAX)
		ss<< "never";
	else
		ss<< _param.stopTrainingCost;
	ss>>stopTrainingCost;
	cout<< "sampleType       : " << setw(10) <<_param.sampleType       << "  trainType        : " << _param.trainType << endl
		<< "stopTrainingCost : " << setw(10) <<stopTrainingCost        << "  trainStart       : " << _param.trainStart << endl
		<< "trainFeature     : " << setw(10) <<_param.trainFeature     << "  trainEnd         : " << _param.trainEnd << endl
		<< "costFunction     : " << setw(10) <<_param.costFunction     << "  trainNumber      : " << _param.trainNumber << endl
		<< "iteration        : " << setw(10) <<_param.iteration        << "  testType         : " << _param.testType << endl
		<< "learningRate     : " << setw(10) <<_param.learningRate     << "  testStart        : " << _param.testStart << endl
		<< "hidden.size()    : " << setw(10) <<_param.hidden.size()    << "  testEnd          : " << _param.testEnd << endl
		<< "output           : " << setw(10) <<_param.output.nodes     << "  testNumber       : " << _param.testNumber << endl

		<< "featureOffset    : " << setw(10) <<_param.featureOffset << endl

		<< "normalizeMethod  : " << setw(10) <<_param.normalizeMethod  << "  testStep         : " << _param.testStep << endl
		<< "loadWeight       : " << setw(10) <<_param.loadWeight       << "  saveWeight       : " << _param.saveWeight << endl
		<< "weightPath       : " << setw(10) <<_param.weightPath       << "  weightName       : " <<_param.weightName << endl
		<< "defaultActivation: " << setw(10) <<_param.defaultActivation << endl
		<< "sampleData       : " << setw(10) <<_param.sampleData << endl;


}
*/
/*
void nn::showd(){
	int i;
	cout << "----------" << "-input--" << "----------" << endl;
	cout<< " input feature : "<< layer[0].o.n_cols-1
		<< " +1 bias" << endl;
	for(i=1;i<(int)layer.size()-1;++i){
	cout << "----------" << "hidden"<<setw(2)<< i << "----------" << endl;
	cout<< " input : "<< layer[i].w.n_rows
		<< " nodes : "<< layer[i].w.n_cols << endl
		<< " weight : " << layer[i].w
		<< " del : " << layer[i].del
		<< " o : " << layer[i].o << endl;
	}
	cout << "----------" << "-output-" << "----------" << endl;
	cout<< " input : "<< layer[i].w.n_rows
		<< " nodes : "<< layer[i].w.n_cols << endl
		<< " weight : " << layer[i].w
		<< " del : " << layer[i].del
		<< " o : " << layer[i].o << endl;

	cout << "----------" << "--------" << "----------" << endl;
}
*/
