#include "neural.hpp"
#include "stringCheck.hpp"
#include "nnio.hpp"
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace arma;

double sigmoid(double in){
	return 1/(1+exp(-1*in));
}

double dsigmoid(double in){
	double re = sigmoid(in);
	return re*(1-re);
}
/*
double tanh(double in){
	return tanh(in);
}
*/
double dtanh(double in){
	double re = tanh(in);
	return 1-re*re;
}

nn::nn(const string& path){
	_init = true;
	act = sigmoid;
	dact = dsigmoid;
	if( !load(path) )
		_init = false;
}

bool nn::readSample(const string& path){
	sample s;
	bool loadsuccess;
	if(_type == nn_t::timeseries)
		loadsuccess = s.read(path.c_str(),true);
	else
		loadsuccess = s.read(path.c_str());
	if( loadsuccess ){
		//s.list();
		_n_feature = s.n_feature();
		_n_sample = s.size();

		//if(_type == nn_t::classification){
		if(true){
			outputs = zeros<rowvec>(_n_sample);
			for(int i=0; i<_n_sample; ++i){
				features.push_back( zeros<rowvec>( _n_feature ) );
				//outputs(i) = s[i].label[0] * outputScale;
				outputs(i) = s[i].label[0];
				for(int j=0; j<_n_feature; ++j)
					features[i](j)= s[i].feature[j];
			}
			nn_a::getNormalizeParam_o(outputs,_outputNormParam);
			nn_a::normalize_o(outputs,_outputNormParam);

			nn_a::getNormalizeParam(features,_featureNormParam);
			nn_a::normalize(features,_featureNormParam);
		}
		return true;
	}

	return false;
}

bool nn::load(const string &path){
	if(!readnn(path, _param)) return false;
	if(!enableParam()) return false;
	show();
	return true;
}

bool nn::enableParam(){
	_type = _param.sampleType;

	iteration = _param.iteration;
	learningRate = _param.learningRate;
	if(!readSample(_param.sampleData))
		return false;
	if(_param.samplingNumber == -1)
		_param.samplingNumber = _n_sample;

	if(_param.testSampleNumber == -1)
		_param.testSampleNumber = _n_sample;

	if(_param.testSampleEnd == -1)
		_param.testSampleEnd = _n_sample;

	nlayer li(nlayer::input, _n_feature);
	if( !li.success() ) return false;
	layer.push_back( li );

	sort(_param.hidden.begin(), _param.hidden.end());

	if( (int)_param.hidden.size() != _param.hidden.back().level )
		return false;
	nlayer *lh;
	for(int i=1; i<=(int)_param.hidden.size(); ++i){
		if(_param.hidden[i-1].activation == "sigmoid")
		lh = new nlayer(nlayer::hidden,
				layer[i-1].o.n_cols,
				_param.hidden[i-1].nodes,
				sigmoid, dsigmoid);
		else if(_param.hidden[i-1].activation == "tanh")
		lh = new nlayer(nlayer::hidden,
				layer[i-1].o.n_cols,
				_param.hidden[i-1].nodes,
				tanh, dtanh);
		else if(_param.hidden[i-1].activation == "")
		lh = new nlayer(nlayer::hidden,
				layer[i-1].o.n_cols,
				_param.hidden[i-1].nodes,
				act, dact);
		else
			errorString("no such activation",_param.hidden[i-1].activation,"");
		if(!lh->success()) return false;
		layer.push_back( *lh );
	}

	nlayer oh(nlayer::output, layer[layer.size()-1].o.n_cols, 1, act, dact);
	if(!oh.success()) return false;
	layer.push_back( oh );
	return true;
}

void nn::show(){
	int i;
	cout << "----------" << "-input--" << "----------" << endl;
	cout<< " input feature : "<< layer[0].o.n_cols-1
		<< " +1 bias" << endl;
	for(i=1;i<(int)layer.size()-1;++i){
	cout << "----------" << "hidden"<<setw(2)<< i << "----------" << endl;
	cout<< " input : "<< layer[i].w.n_rows
		<< " nodes : "<< layer[i].w.n_cols
		<< " nodes : "<< _param.hidden[i-1].activation << endl;
	}
	cout << "----------" << "-output-" << "----------" << endl;
	cout<< " input : "<< layer[i].w.n_rows
		<< " nodes : "<< layer[i].w.n_cols << endl;

	cout << "----------" << "--------" << "----------" << endl;
}

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

inline
void nn::setInput(arma::rowvec &input){
	for(int i=0; i<_n_feature; ++i)
		layer[0].o(i) = input(i);
}

inline
void nn::setInputSeries(int s){
	for(int i=0; i<_n_feature; ++i)
		layer[0].o(i) = features[s+i](0);
}

int nn::getFirstTestSample(){
	switch(_param.testSampleType){
	case nn_t::all :
			return 0;
		break;
	case nn_t::number :
			return _param.testSampleStart;
		break;
	}
	return -1;
}

inline
bool nn::getNextTestSample(int &sample){
	static int counter = _param.testSampleStart+1;
	static nn_t::testSample_t testSampleType = _param.testSampleType;
	static int testSampleStart	= _param.testSampleStart;
	static int testSampleEnd	= _param.testSampleEnd;
	switch(testSampleType){
	case nn_t::all :
		sample = counter;
		if( counter ==  _n_sample){
			counter = 1;
			sample = 0;
			return false;
		}
		++counter;
		break;
	case nn_t::number :
		sample = counter;
		if( counter == testSampleEnd){
			counter = testSampleStart;
			return false;
		}
		++counter;
		break;
	}
	return true;
}

int nn::getFirstSample(){
	switch(_param.samplingType){
	case nn_t::all :
			return 0;
		break;
	case nn_t::number :
			return _param.samplingStart-1;
		break;
	case nn_t::bunch :
			return 0;
		break;
	}
	return -1;
}

inline
bool nn::getNextSample(int &sample){
	static int counter = _param.samplingStart;
	static nn_t::sampling_t samplingType = _param.samplingType;
	static int samplingStart = _param.samplingStart;
	static int samplingEnd = _param.samplingEnd;
	static int samplingNumber = _param.samplingNumber;
	static int bunch_counter = 1;

	switch(samplingType){
	case nn_t::all :
		sample = counter;
		if( counter ==  _n_sample){
			counter = 1;
			sample = 0;
			return false;
		}
		++counter;
		break; case nn_t::number :
		sample = counter;
		if( counter == samplingEnd){
			counter = samplingStart-1;
			sample = counter;
			++counter;
			return false;
		}
		++counter;
		break;
	case nn_t::bunch :
		sample = counter;

		if( counter == _n_sample || counter == samplingEnd){
			counter = 1;
			bunch_counter = 1;
			sample = 0;
			return false;
		}
		if(	counter % samplingNumber == 0){
			counter = bunch_counter * samplingNumber+1;
			sample = counter-1;
			++bunch_counter;
			return false;
		}
		++counter;
		break;
	}
	return true;
}

void nn::test(){
	//forward
	int nodes;
	int i;
	for(i=1; i<(int)layer.size()-1; ++i){
		layer[i].s = layer[i-1].o * layer[i].w;
		nodes = layer[i].n_nodes()-1;
		for(int j=0; j<nodes; ++j){
			layer[i].o(j) = layer[i].act(layer[i].s(j));
		}
	}
	layer[i].s = layer[i-1].o * layer[i].w;
	nodes = layer[i].n_nodes();
	for(int j=0; j<nodes; ++j){
		layer[i].o(j) = layer[i].act(layer[i].s(j));
	}
	//layer[i].show();
}

void nn::test(int sample){
	setInput(features[sample]);
	//forward
	int nodes;
	int i;
	for(i=1; i<(int)layer.size()-1; ++i){
		layer[i].s = layer[i-1].o * layer[i].w;
		nodes = layer[i].n_nodes()-1;
		for(int j=0; j<nodes; ++j){
			layer[i].o(j) = layer[i].act(layer[i].s(j));
		}
	}
	layer[i].s = layer[i-1].o * layer[i].w;
	nodes = layer[i].n_nodes();
	for(int j=0; j<nodes; ++j){
		layer[i].o(j) = layer[i].act(layer[i].s(j));
	}
	//layer[i].show();
}


inline
void nn::clear_dels(){
	layer.back().es.zeros();
	for(int i=1; i<(int)layer.size(); ++i)
		layer[i].dels.zeros();
}

inline
void nn::cal_dels(double label){
	//output to hidden
	nlayer& o = layer.back();
	o.e(0) = label - o.o(0);
	o.d(0) = o.e(0) * o.dact(o.s(0));
	for(int j=0; j<layer[layer.size()-2].n_nodes(); ++j){
		o.del(j,0) =
			learningRate * o.d(0) * layer[layer.size()-2].o(j);
	}
	o.dels += o.del;
	o.es(0) += 0.5*o.e(0)*o.e(0);
	//hidden to input
	int nodes, nodes_;
	for(int l = layer.size()-2; l>0; --l){
		nlayer& nl = layer[l];
		nlayer& nl_ = layer[l+1];
		nodes = nl.n_nodes();
		nodes_= nl_.n_nodes();

		nl.d.zeros();
		for(int i=0; i<nodes; ++i){
			for(int j=0; j<nodes_; ++j){
				nl.d(i) += nl_.d(j) * nl_.w(i,j);
			}
			nl.d(i) *= nl.dact(nl.s(i));
			for(int j=0; j<nl.n_input(); ++j)
			{
				nl.del(j,i) =
					learningRate * nl.d(i) * layer[l-1].o(j);
			}
		}
		nl.dels += nl.del;
	}
}


inline
void nn::wupdate(){
	static int samplingNumber = _param.samplingNumber;
	//cout << "samplingNumber " << samplingNumber<<endl;
	for(int i=1; i<(int)layer.size(); ++i)
		layer[i].w += layer[i].dels;
		//layer[i].w += layer[i].dels/samplingNumber;
}

void nn::error(int &i){
	static int ep = ceil((double)iteration/1000);
	static double last_t = clock();
	static double last_c = 0;
	static int stop_counter = 0;
	double errs = 0;
	bool show = ( ( clock() - last_t ) >= 500000);
	bool save = ( i%ep == 0 );
	if(show || save){
		for(int j=0; j<layer.back().n_nodes(); ++j){
			errs += layer.back().es(j)/_param.samplingNumber;
		}
		if( abs(errs - last_c) < _param.stopTrainingCost)
			stop_counter++;
		else
			stop_counter =0;

		if(stop_counter==10)
		{
			i = iteration;
			cout << endl <<" cost limit , stop training "<<endl;
		}
	}
	if( show ){
		cout.flush();
		cout<< '\r'<< " iteration : " <<  setw(7) << i
			<<" average cost : " << setw(10) << errs
			<<" cost rate : " << setw(10) << abs(errs-last_c)
			<<" , " << (errs-last_c)*100/errs << "%    " ;
		last_t = clock();
		last_c = errs;
	}
	if( save ){
		e.push_back(errs);
	}
}

void nn::train(){
	/*
	for(int i=0; i<_n_sample; ++i)
		cout << outputs(i) << " : " << features[i];
		*/
	int s = getFirstSample();
	for(int i=0; i<iteration; ++i){
		clear_dels();
		do{
			/*
			showd();
			cin.get();
			*/
			//forward
			if( _type == nn_t::timeseries){
				cout <<"gg"<<i<<":" <<s<< endl;
				setInputSeries(s);
				test();
				cal_dels(features[s+1](0));
			}
			else{
				//cout << i <<":" << s << endl;
				test( s );
				cal_dels(outputs[s]);
			}
			//back propagation
		}while(getNextSample(s));
		error(i);
		wupdate();
		if(i%1000 ==1000)
			testResult();
	}
	cout << endl;
}


void nn::testResultRegression(){
	double errors=0, singleError;
	int s = getFirstTestSample();
	int step = _param.testStep;
	if(step == 0)
		step = _n_sample;
	int j=0;
	do{
		test( s );
		singleError = 0.5* pow(outputs( s )-layer.back().o(0), 2);
		errors += singleError;
		if(j<step)
			j++;
		else{
			cout<< " feature : " << features[s]
				<< " output : " << layer.back().o(0) << endl
				<< " desire : " << outputs(s) << endl;
			cin.get();
			j=0;
		}
	}while(getNextTestSample(s));
	cout<< " average cost " << errors/_param.testSampleNumber << endl;

}

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
void nn::testResult(){
	int s = getFirstTestSample();
	int step = _param.testStep;
	int outputClass, realClass;
	int errors=0;
	if(step == 0)
		step = _n_sample;

	int j=0;
	do{
		test( s );
		realClass =
			round( ( ( outputs( s )-0.5 )
					 /_outputNormParam.scale(0)		)
				+_outputNormParam.average(0) );
		outputClass =
			round( ( ( layer.back().o(0)-0.5)
					/_outputNormParam.scale(0)		)
					+_outputNormParam.average(0));
		if(outputClass != realClass)
			errors++;
		if(j<step)
			j++;
		else{
			cout<< " feature : " << features[s]
				<< " output : " << layer.back().o(0) << ":"<< outputClass<< endl
				<< " real : " << outputs(s) <<":"<<realClass<< endl;
			cin.get();
			j=0;
		}
	}while(getNextTestSample(s));
	cout<< " all : "<< _param.testSampleNumber <<" error : " << errors << " predict rate "<< (1-(double)errors/_param.testSampleNumber)*100<<'%' << endl;

}
