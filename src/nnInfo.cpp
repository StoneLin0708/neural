#include "neural.hpp"
#include <iostream>
#include <iomanip>
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
	cout<< "sampleType       : " << _param.sampleType << endl
		<< "stopTrainingCost : " << _param.stopTrainingCost << endl
		<< "trainFeature     : " << _param.trainFeature << endl
		<< "sampleData       : " << _param.sampleData << endl
		<< "iteration        : " << _param.iteration << endl
		<< "learningRate     : " << _param.learningRate << endl
		<< "hidden.size()    : " << _param.hidden.size() << endl
		<< "output           : " << _param.output.nodes << endl

		<< "normalizeMethod  : " << _param.normalizeMethod << endl
		<< "loadWeight       : " << _param.loadWeight << endl
		<< "saveWeight       : " << _param.saveWeight << endl
		<< "weightPath       : " << _param.weightPath << endl

		<< "weightName       : " << _param.weightName << endl
		<< "defaultActivation: " << _param.defaultActivation << endl
		<< "featureOffset    : " << _param.featureOffset << endl

		<< "trainType        : " << _param.trainType << endl
		<< "trainStart       : " << _param.trainStart << endl
		<< "trainEnd         : " << _param.trainEnd << endl
		<< "trainNumber      : " << _param.trainNumber << endl

		<< "testType         : " << _param.testType << endl
		<< "testStart        : " << _param.testStart << endl
		<< "testEnd          : " << _param.testEnd << endl
		<< "testNumber       : " << _param.testNumber << endl

		<< "testStep         : " << _param.testStep << endl
		<< "costFunction     : " << _param.costFunction << endl;

}
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
