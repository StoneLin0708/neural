#include "neural.hpp"
#include "stringCheck.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace arma;
/*
nlayer::nlayer(){
	_init = false;
}
*/
nlayer::nlayer(
		layer_t type, int n_input, int n_nodes,
		double (*act)(double in),
		double (*dact)(double in) ){
	if( type != input &&
		n_input < node_max &&
		n_input >= 1 &&
		n_nodes < node_max &&
		n_nodes >= 1 &&
		act != NULL &&
		dact != NULL  ){
		_type = type;
		_initialize(type, n_input, n_nodes,
				act,dact);
		_init = true; }
	else if(type == input &&
		n_input < node_max &&
		n_input > 1){
		_type = type;
		_initialize_i(n_input);
		//cout<< "inputl: " <<_nodes<<endl;
		_init = true;
		}
	else{
		cout<< "fail initialize layer with parameter : " << endl
			<< " type    : " << type << endl
			<< " n_nodes : " << n_input << endl
			<< " n_nodes : " << n_nodes << endl
			<< " act     : " << act << endl
			<< " dact    : " << dact << endl;
	}
}

void nlayer::_initialize(
		layer_t type, int n_input, int n_nodes,
		double (*act)(double in),
		double (*dact)(double in) ){
	_input = n_input;
	_nodes = n_nodes + type;
	random_w(-0.5, 0.5);
	s = zeros<rowvec>(_nodes);
	o = zeros<rowvec>(_nodes);
	if(type == output){
		e = zeros<rowvec>(_nodes);
		es = zeros<rowvec>(_nodes);
	}
	d = zeros<rowvec>(_nodes);
	del = zeros<mat>(_input, _nodes);
	dels = zeros<mat>(_input, _nodes);
	this->act = act;
	this->dact = dact;
	o(_nodes-1) = 1;
}

void nlayer::_initialize_i(int n_input){
	_nodes = n_input+1;
	o = zeros<rowvec>(_nodes);
	o(_nodes-1) = 1;
}

void nlayer::random_w(double min, double max){
	w = randu(_input, _nodes)*(max-min);
	mat mmin(_input, _nodes);
	mmin.fill(min);
	w += mmin;
}

void nlayer::show(){
	cout<< "----------type"<< _type <<"----------" << endl;
	cout<< "w" << endl << w << endl
		<< "s" << endl << s << endl
		<< "o" << endl << o << endl
		<< "d" << endl << d << endl
		<< "del" << endl << del << endl
		<< "dels" << endl << dels << endl;
}
//-------------------------------------------------------------------
/*
nn::nn(int input_number, int hidden_number, int output_number){
	randomInit(); //random hidden & output weight
	learning_rate = dlearning_rate;
	normalize_scale = 1.0;
}
*/
nn::nn(string& path, double (*activation)(double), double (*dactivation)(double)){
	_init = true;
	act = activation;
	dact = dactivation;
	if( !readnn(path) )
		_init = false;
}

bool nn::readSample(const string& path){
	sample s;
	if( s.read(path.c_str()) ){
		//s.list();
		n_feature = s.n_feature();
		n_label = s.n_label();
		n_sample = s.size();

		if(type == multiOutput){
			for(int i=0; i<n_sample; ++i){
				mlabels.push_back( zeros<rowvec>( n_label ) );
				features.push_back( zeros<rowvec>( n_feature ) );
				for(int j=0; j<n_label; ++j)
					mlabels[i](j) = s[i].label[j] * output_scale;
				for(int j=0; j<n_label; ++j)
					features[i](j) = s[i].feature[j] * normalize_scale;
			}
		}
		else if(type == singleOutput){
			if(n_label != 1){
				cout << " label not single " << endl;
				return false;
			}
			slabels = zeros<rowvec>(n_sample);
			for(int i=0; i<n_sample; ++i){
				features.push_back( zeros<rowvec>( n_feature ) );
				slabels(i) = s[i].label[0] * output_scale;
				for(int j=0; j<n_feature; ++j)
					features[i](j)= s[i].feature[j] * normalize_scale;
			}
		}
		/*
		cout << slabels << endl;
		for(int i=0; i<mlabels.size(); ++i)
			cout << mlabels[i] << endl;
		for(int i=0; i<n_sample; ++i)
			cout << features[i] << endl;
			*/
		return true;
	}
	else{
		return false;
	}
};

bool nn::readLayer(string& in){
	int nodes;
	string out;
	nlayer::layer_t type;

	if(readFor("hidden=",in,out)){
		type = nlayer::hidden;
	}
	else if(readFor("output=",in,out)){
		type = nlayer::output;
	}
	else{
		errorString(" error layer type ", in, "output= || hidden=");
		return false;
	}

	vector<string> sout = split(out,',');

	if( sout.size() != 2 ){
		errorString(" error layer argument ", out,
				"layer number, node number");
		return false;
	}

	if(!isInt(sout[0]))
		return false;
	if( (int)layer.size() != atoi(sout[0].c_str()) ){
		errorString(" error layer ", sout[0], "");
		return false;
	}

	if(!isInt(sout[1]))
		return false;
	nodes = atoi(sout[1].c_str());

	nlayer nl(type, layer[layer.size()-1].n_nodes(), nodes, act, dact);
	if(!nl.success())
		return false;

	layer.push_back(nl);

	return true;
}

bool nn::readnn(const string& path){
    ifstream fnn;
    fnn.open(path.c_str(), ios::in);
	string in,out;

    if ( fnn.fail() ){
        cout<< "fail to open nn file : " << path<< endl;
        return false;
    }
	int line=0;
    while( fnn >> in ){
		cout << " readnn : " << in << endl;
		line++;
		switch(line){
		case 1:
			if( !readFor("sampleType=",in,out) )
				return false;
			if( out == "singleOutput" )
				type = singleOutput;
			else if( out == "multiOutput")
				type = multiOutput;
			else{
				errorString(" tpye wrong ", out ," singleOutput || multiOutput");
				return false;
			}
			break;
		case 2:
			if( !readFor("sampleScale=",in,out) )
				return false;
			if( !isFloat(out))
				return false;
			normalize_scale = (double)atof(out.c_str());
			break;
		case 3:
			if( !readFor("outputScale=",in,out) )
				return false;
			if( !isFloat(out))
				return false;
			output_scale = (double)atof(out.c_str());
			break;
		case 4:
			if( !readFor("sampleData=",in,out) )
				return false;
			for(int i=0; i<(int)out.size()-1; ++i)
				out[i] = out[i+1];
			out.resize(out.size()-2);
			//cout << out << endl;
			if(!readSample(out))
				return false;
			layer.push_back( nlayer(nlayer::input, n_feature) );
			break;
		case 5:
			if( !readFor("iteration=",in,out) )
				return false;
			if( !isInt(out) )
				return false;
			iteration = atoi(out.c_str());
			break;
		case 6:
			if( !readFor("learningRate=",in,out) )
				return false;
			if( !isFloat(out) )
				return false;
			learning_rate = (double)atof(out.c_str());
			break;
		default :
			if( !readLayer(in) )
				return false;
			break;
		}
    }

    fnn.close();
	if( layer.back().type() != nlayer::output){
		cout << "nn initialize fail : no output layer" << endl;
		return false;
	}
	for(int i=1; i<(int)layer.size()-1; ++i){
		if(layer[i].type() != nlayer::hidden){
			cout << "nn initialize fail : output layer more than one" << endl;
			return false;
		}
	}

	show();
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
		<< " nodes : "<< layer[i].w.n_cols << endl;
	}
	cout << "----------" << "-output-" << "----------" << endl;
	cout<< " input : "<< layer[i].w.n_rows
		<< " nodes : "<< layer[i].w.n_cols << endl;

	cout << "----------" << "--------" << "----------" << endl;
}

inline
void nn::setInput(arma::rowvec &input){
	for(int i=0; i<n_feature; ++i)
		layer[0].o(i) = input(i);
}

inline
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
	//for(int i=0; i<o.n_nodes(); ++i){
		o.e(0) = label - o.o(0);
		o.d(0) = o.e(0) * o.dact(o.s(0));
		for(int j=0; j<layer[layer.size()-2].n_nodes(); ++j){
			o.del(j,0) =
				learning_rate * o.d(0) * layer[layer.size()-2].o(j);
		}
		o.dels += o.del;
		o.es += o.e;
	//}
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
					learning_rate * nl.d(i) * layer[l-1].o(j);
			}
		}
		nl.dels += nl.del;
	}

}
inline
void nn::cal_delm(arma::rowvec &label){
	//output to hidden
	nlayer& o = layer.back();
	for(int i=0; i<o.n_nodes(); ++i){
		o.e(i) = label(i) - o.o(i);
		o.d(i) = o.e(i) * o.dact(o.s(i));
		for(int j=0; j<layer[layer.size()-2].n_nodes(); ++j){
			o.del(j,i) =
				learning_rate * o.d(i) * layer[layer.size()-2].o(j);
		}
		o.dels += o.del;
		o.es += o.e;
	}
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
					learning_rate * nl.d(i) * layer[l-1].o(j);
			}
		}
		nl.dels += nl.del;
	}

}

inline
void nn::wupdate(){
	for(int i=1; i<(int)layer.size(); ++i)
		layer[i].w += layer[i].dels;
}

void nn::error(int i){
	double errs = 0;
	for(int j=0; j<layer.back().n_nodes(); ++j){
		errs+= abs(layer.back().es(j));
	}
	cout<< '\r'<< "iteration : " <<  setw(7) << i
		<<" error : " << setw(10) << errs;
	e.push_back(errs);
}

void nn::train(){
	int ep = iteration/1000;
	if(ep == 0 ) ep=1;
	//for(int i=0; i<1; ++i){
	for(int i=0; i<iteration; ++i){
		clear_dels();
		//for(int s=0; s<2; ++s){
		for(int s=0; s<n_sample; ++s){
			//set input
			setInput( features[s] );
			//cout << layer[0].o << endl;
			//forward
			test();
			//cout << layer[2].o << endl;
			//back propagation
			if(type == singleOutput)
				cal_dels(slabels[s]);
			else
				cal_delm(mlabels[s]);
			/*
			cout << layer[1].w << endl;
			cout << layer[1].del << endl;
			cout << layer[2].w << endl;
			cout << layer[2].del << endl;
			*/
		}
		//if(false){
		if( i%ep == 0){
			error(i);
		}
		//finish a iteration, change neural weight
		wupdate();
	}
	cout << endl;
}

