#include "neural.hpp"
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
		n_input > 1 &&
		n_nodes < node_max &&
		n_nodes > 1 &&
		act != NULL &&
		dact != NULL  ){
		_type = type;
		_initialize(type, n_input, n_nodes,
				act,dact);
		cout<< "input : " <<_input<<endl;
		cout<< "nodes : " <<_nodes<<endl;
		_init = true;
	}
	else if(type == input &&
		n_input < node_max &&
		n_input > 1){
		_type = type;
		_initialize_i(n_input);
		cout<< "inputl: " <<_nodes<<endl;
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
	input_num = input_number;
	hidden_num = hidden_number;
	output_num = output_number;
	input = zeros(input_num+1);

	randomInit(); //random hidden & output weight

	hs = zeros(hidden_num); //hidden out
	ho = zeros(hidden_num+1); //hidden activated

	os = zeros(output_num); //output out
	oo = zeros(output_num); //output activated
	de = zeros(output_num); //output activated

	input.at(input_num) = 1; //bias
	ho.at(hidden_num) = 1; //bias

	hdel = zeros(input_num+1, hidden_num+1); //hidden delta
	odel = zeros(hidden_num+1, output_num); //output delta
	od = zeros(output_num);
	hd = zeros(hidden_num+1);
	hdels = zeros(input_num+1, hidden_num+1); //hidden delta
	odels = zeros(hidden_num+1, output_num); //output delta
	ods = zeros(output_num);
	hds = zeros(hidden_num+1);

	oerr = zeros(output_num);
	oerrs = zeros(output_num);

	learning_rate = dlearning_rate;
	normalize_scale = 1.0;
}
*/
nn::nn(std::string& path, double (*activation)(double), double (*dactivation)(double)){
	act = activation;
	dact = dactivation;
	readnn(path);
	de = zeros<rowvec>(layer[layer.size()-1].n_nodes());
}

sample& nn::getSample(){
	return _s;
}

bool nn::readSample(std::string& path){
	if( _s.read(path.c_str()) ){
		_s.list();
		return true;
	}
	else{
		return false;
	}
};

bool nn::readFor(int line, string& in, const string text, bool test){
	int i=0;
	string tmp;
	while(in[i] == text[i]){
		//cout<< in[i] << " : "<< text[i] << endl;
		if((int)text.size() == i+1){
			for(int j=in.size()-1; j>=(int)text.size(); --j){
				tmp.push_back(in[j]);
			}
			in.clear();
			for(int j=tmp.size()-1; j>=0; --j){
				in.push_back(tmp[j]);
			}
			cout<<" read " << text << " : "<< in << endl;
			return true;
		}
		i++;
	}
	if(!test)
	cout<< "nn read fail at " << line
		<< " : "<< in << " : " << in[i] << " : " << text[i] << endl;
	return false;
}

void nn::errString(std::string& line, std::string& str, int s ,int e){
	cout<< " neural error in line  \"" << line << "\""<< endl
		<< "  at character " << s << " to " << e
		<< " \""<< str << "\""<< endl;
}

bool nn::readLayer(int line, std::string& in){
	int s,e;
	string tmp;
	nlayer::layer_t type;
	if(readFor(line, in, "hidden=",true)){
		type = nlayer::hidden;
	}
	else if(readFor(line, in, "output=")){
		type = nlayer::output;
	}
	else
		return false;

	s = 0;
	e = in.find(',',s);
	if(e == (int)string::npos){
		errString(in, tmp, s, (int)in.size());
		return false;
	}
	for(int i=s; i<e; ++i)
		tmp.push_back(in[i]);
	if(!_s.isInt(tmp)){
		errString(in, tmp, s, e);
		return false;
	}
	if((int)layer.size() != atoi(tmp.c_str()))
		return false;
	tmp.clear();
	s = e+1;
	e = in.size();
	if(e == (int)string::npos){
		errString(in, tmp, s, (int)in.size());
		return false;
	}
	for(int i=s; i<e; ++i)
		tmp.push_back(in[i]);
	if(!_s.isInt(tmp)){
		errString(in, tmp, s, e);
		return false;
	}

	nlayer nl(type, layer[layer.size()-1].n_nodes(), atoi(tmp.c_str()), act, dact);
	if(!nl._init){
		cout<<"nl init fail line "<<line<<" : "<<in<<endl;
		return false;
	}
	else
		//nl.show();
	layer.push_back(nl);

	return true;
}

bool nn::readnn(std::string& path){
    ifstream fnn;
    fnn.open(path.c_str(), ios::in);
	string in;

    if ( fnn.fail() ){
        cout<< "fail to open nn file : " << path<< endl;
        return false;
    }
	int line=0;
    while( fnn >> in ){
		//cout << "in = " << in << endl;
		line++;
		switch(line){
		case 1:
			if( readFor(line, in, "sampledata=") ){
				for(int i=0; i<(int)in.size()-1; ++i)
					in[i] = in[i+1];
				in.resize(in.size()-2);
				cout << in << endl;
				if(!readSample(in))
					return false;
			}
			else
				return false;
			break;
		case 2:
			if( readFor(line, in, "samplescale=") ){
				if(_s.isFloat(in))
					normalize_scale = (double)atof(in.c_str());
				else{
					cout<<" data wrong ! "<< line<< " : "<< in<< endl;
					return false;
				}
			}
			else
				return false;
			break;
		case 3:
			if( readFor(line, in, "iteration=") ){
				if(_s.isInt(in))
					iteration = atoi(in.c_str());
				else{
					cout<<" data wrong ! "<< line<< " : "<< in<< endl;
					return false;
				}
			}
			else
				return false;
			break;
		case 4:
			if( readFor(line, in, "learningrate=") ){
				if(_s.isFloat(in))
					learning_rate = (double)atof(in.c_str());
				else{
					cout<<" data wrong ! "<< line<< " : "<< in<< endl;
					return false;
				}
			}
			else
				return false;
			break;
		case 5:
			if( readFor(line, in, "inputfeature=") ){
				if(_s.isInt(in)){
					input_num = atoi(in.c_str());
					layer.push_back(
							*new nlayer(nlayer::input, input_num) );
					//layer[0].show();
				}
				else{
					cout<<" data wrong ! "<< line<< " : "<< in<< endl;
					return false;
				}
			}
			else
				return false;
			break;
		default :
			if( !readLayer(line, in) )
				return false;
			break;
		}
    }

    fnn.close();
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

void nn::clear_dels(){
	layer.back().es.zeros();
	for(int i=1; i<(int)layer.size(); ++i)
		layer[i].dels.zeros();
}

void nn::cal_del(){
	//output to hidden
	nlayer& o = layer.back();
	for(int i=0; i<o.n_nodes(); ++i){
		o.e(i) = de(i) - o.o(i);
		o.d(i) = o.e(i) * o.dact(o.s(i));
		for(int j=0; j<layer[layer.size()-2].n_nodes(); ++j){
			o.del(j,i) =
				learning_rate * o.d(i) * layer[layer.size()-2].o(j);
		}
		o.dels += o.del;
		o.es += o.e;
		//o.show();
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

void nn::wupdate(){
	for(int i=1; i<(int)layer.size(); ++i)
	{
		layer[i].w += layer[i].dels;
		//layer[i].show();
	}
}

void nn::error(int i){
	double errs = 0;
	for(int j=0; j<layer.back().n_nodes(); ++j){
		errs+= abs(layer.back().es(j));
	}
	system("setterm -cursor off");
	cout<< '\r'<< "iteration : " <<  setw(7) << i
		<<" error : " << setw(10) << errs;
	system("setterm -cursor on");
	e.push_back(errs);
}

void nn::train(){
	int fnum = (int)_s[0].feature.size();

	for(int i=0; i<iteration; ++i){
		clear_dels();
		//for(int s=0; s<1; ++s){
		for(int s=0; s<(int)_s.size(); ++s){
			//set input
			//get sample performance is poor,need rework
			for(int j=0; j<fnum; ++j)
				layer[0].o(j) = _s[s].feature[j]*normalize_scale;
			//set desire output
			de.fill(0);
			de( _s[s].l ) = 1;
			//forward
			test();
			//back propagation
			cal_del();
		}
		if(i%50 == 0){
		//if(false){
			error(i);
		}
		//finish a iteration, change neural weight
		wupdate();
	}
	cout << endl;
}

