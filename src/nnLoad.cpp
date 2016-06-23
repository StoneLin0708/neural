#include "neural.hpp"
#include "nnfun.hpp"
#include <stringProcess.hpp>
#include <math.h>

using arma::zeros;
using std::cout;
using std::endl;

bool nn::readSample(const string& path){
	sample s;
	if(_type == nn_t::timeseries){
		if(!s.read(path.c_str(),true)) return false;
		_n_sample = s.size();
		_n_feature = s.n_feature();
		_n_output = 1;
		_param.featureOffset = 1;
		features = zeros<rowvec>( _n_sample );
		for(int i=0; i<_n_sample; ++i)
				features( i ) = s[i].feature[0];
		featureNormParam = nn_a::normalize(features, _n_feature, -0.8, 0.8);
		_n_sample = s.size()-_param.trainFeature;
		return true;
	}else if(_type == nn_t::classification){
		if(!s.read(path.c_str())) return false;
		_n_sample = s.size();
		_n_feature = s.n_feature();
		_n_output = s.n_label();
		if(_n_output != 1){
			cout << "not a classification problem" << endl;
			return false;
		}
		int classes = -1;
		for(int i=0; i<_n_sample; ++i){
			if(classes < s[i].label[0])
				classes = s[i].label[0];
		}
		_n_output = classes+1;

		features = zeros<rowvec>(_n_feature * _n_sample);
		outputs = zeros<rowvec>(_n_output * _n_sample);
		for(int i=0; i<_n_sample; ++i){
			for(int j=0; j<_n_feature; ++j)
				features(i*_n_feature + j) = s[i].feature[j];
			outputs( i*_n_output + round(s[i].label[0]) ) = 1.0;
		}

		featureNormParam = nn_a::normalize(features, _n_feature, -1, 1);
		outputNormParam = nn_a::normalize(outputs, _n_output, 0.01, 0.99);
		_param.trainFeature = _n_feature;
		_param.featureOffset = _n_feature;
	//	_param.numberOfOutput = _n_output;
		return true;
	}else{
		if(!s.read(path.c_str())) return false;
		_n_sample = s.size();
		_n_feature = s.n_feature();
		_n_output = s.n_label();
		features = zeros<rowvec>(_n_feature * _n_sample);
		outputs = zeros<rowvec>(_n_output * _n_sample);

		for(int i=0; i<_n_sample; ++i){
			for(int j=0; j<_n_feature; ++j)
				features(i*_n_feature + j) = s[i].feature[j];
			for(int j=0; j<_n_output; ++j)
				outputs(i*_n_output + j) = s[i].label[j];
		}
		featureNormParam = nn_a::normalize(features, _n_feature, -1, 1);
		outputNormParam = nn_a::normalize(outputs, _n_output, 0, 1);
		_param.trainFeature = _n_feature;
		_param.featureOffset = _n_feature;
	s.list();
		return true;
	}
}

bool nn::enableParam(){
	iteration = _param.iteration;
	learningRate = _param.learningRate;
//read sample-----------------------------------------
	_type = _param.sampleType;

	if(!readSample(_param.sampleData))
		return false;
	if(_param.trainNumber == 0)
		_param.trainNumber = _n_sample;
	if(_param.testNumber == 0)
		_param.testNumber = _n_sample;
//costFunction----------------------------------------
	if(_param.costFunction == "mse"){
		cost = nn_func::mse;
		dcost = nn_func::dmse;
	}
	else if(_param.costFunction == "nmse"){
		cost = nn_func::nmse;
		dcost = nn_func::dnmse;
	}
	else{
		errorString("param error no costFunction",_param.costFunction,"");
		return false;
	}

//building layer--------------------------------------
	//input
	nnLayerInput li(_param.trainFeature, &features, _param.featureOffset);
	if( !li.success() ) return false;
	Linput = li;
	//hidden
	sort(_param.hidden.begin(), _param.hidden.end());
	if( (int)_param.hidden.size() != _param.hidden.back().level ) return false;

	nnLayerHidden lh;
	int inputn = Linput.n_node();
	double wmin, wmax;
	void (*act)(rowvec &, rowvec& , int);
	void (*dact)(rowvec &, rowvec& , int);

	for(int i=0; i<(int)_param.hidden.size(); ++i){
		if     ( _param.hidden[i].activation == "sigmoid"){
			wmin = -4; wmax = 4; act = nn_funa::sigmoid; dact = nn_funa::dsigmoid;}
		else if( _param.hidden[i].activation == "tanh"){
			wmin = -4; wmax = 4; act = nn_funa::tanh;    dact = nn_funa::dtanh;}
		else if( _param.hidden[i].activation == ""){
			wmin = -4; wmax = 4; act = nn_funa::sigmoid; dact = nn_funa::dsigmoid;}
		else
			errorString("no such activation",_param.hidden[i].activation,"");
		lh = nnLayerHidden(_param.hidden[i].nodes, inputn, wmin, wmax, act, dact);
		if(!lh.success()) return false;
		Lhidden.push_back( lh );
		inputn = Lhidden.back().n_node();
	}
	//output
	if     (_param.output.activation == "sigmoid"){
		wmin = -4; wmax = 4; act = nn_funa::sigmoid; dact = nn_funa::dsigmoid;}
	else if(_param.output.activation == "tanh"){
		wmin = -4; wmax = 4; act = nn_funa::tanh;    dact = nn_funa::dtanh;}
	else if(_param.output.activation == ""){
		wmin = -4; wmax = 4; act = nn_funa::sigmoid; dact = nn_funa::dsigmoid;}
	else
		errorString("no such activation",_param.output.activation,"");

	if(_param.sampleType == nn_t::timeseries)
		Loutput = nnLayerOutput(_param.output.nodes, Lhidden.back().n_node(), wmin, wmax,
				&features, _param.featureOffset, _param.trainFeature, act, dact);
	else
		Loutput = nnLayerOutput(_n_output, Lhidden.back().n_node(), wmin, wmax,
				&outputs, _n_output, 0, act, dact);

	if(!Loutput.success()) return false;

	return true;
	for(int i=0; i<_n_sample; ++i){
		cout << i << " - ";
		Loutput.setOutput( i );
		for(int j=0;j<_n_output; ++j)
			cout << (double)Loutput.desireOut(j) << ',';
		cout << ":";
		Linput.setFeatures( i );
		for(int j=0; j<_n_feature; ++j)
			cout << (double)Linput.out(j) << ',';
		cout << endl;
	}
	return true;
}
