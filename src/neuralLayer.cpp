#include "neural.hpp"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;

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
		dact != NULL  ){ _type = type;
		_initialize(type, n_input, n_nodes,
				act,dact);
		_init = true; }
	else if(type == input &&
		n_input < node_max &&
		n_input > 1){
		_type = type;
		_initialize_i(n_input);
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
	random_w(-4, 4);
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
