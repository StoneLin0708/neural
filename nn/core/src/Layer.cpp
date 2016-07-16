#include "core/include/Layer.hpp"

#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using arma::randu;
using arma::zeros;

namespace nn{

	IOLayer::IOLayer(rowvec *outputs, int offset, int start){

	}

	nnLayerInput::nnLayerInput(){
		_init=false;
	}

	nnLayerInput::nnLayerInput(int node, rowvec *features, int offset){
		if(node < 1 || offset < 1 || (int)features->n_cols < node){
			cout<< "input init fail : " << endl
				<< "node = " << node << endl
				<< "offset = " << offset << endl
				<< "features.n_cols = " << (int)features->n_cols<< endl;
			_init = false;
		}
		else{
			_node = node+1;
			_offset = offset;
			_features = features;

			out.zeros(_node);
			out(_node-1) = 1;
			_init = true;
		}
	}

	void nnLayerInput::setFeatures(int Nth){
		static const int nfeature = _node-1;
		int offsetN = _offset*Nth;
		for(int i=0; i<nfeature; ++i){
			//cout << "in  get:"<<offsetN+i<<endl;
			out(i) = (*_features)( offsetN + i );
		}
	}

	nnLayerHidden::nnLayerHidden(){
		_init = false;
	}

	nnLayerHidden::nnLayerHidden(
			int node, int input,int wmin, int wmax,
			void (*act)(rowvec &in, rowvec &out, int size),
			void (*dact)(rowvec &in, rowvec &out, int size)){
		if(input<1 || node<1 || wmin > wmax){
			cout<< "hidden init fail : " << endl
				<< "node = " << node << endl
				<< "input = " << input << endl
				<< "wmin,wmax = " << wmin << ',' << wmax << endl;
			_init = false;
		}
		else{
			this->act = act;
			this->dact = dact;

			_node = node+1;
			_input = input;

			weight = randu(_input, _node)*(wmax-wmin);
			mat mmin(_input, _node);
			mmin.fill(wmin);
			weight += mmin;

			sum = zeros<rowvec>(_node);
			out = zeros<rowvec>(_node);
			out(_node-1) = 1;

			delta = zeros<rowvec>(_node);
			wupdate = zeros<mat>(_input, _node);
			wupdates = zeros<mat>(_input, _node);
			_init = true;
		}
	}


	nnLayerOutput::nnLayerOutput(){
		_init=false;
	}

	nnLayerOutput::nnLayerOutput(
			int node, int input,int wmin, int wmax,
			rowvec *outputs,int offset, int start,
			void (*act)(rowvec &in, rowvec &out, int size),
			void (*dact)(rowvec &in, rowvec &out, int size)){
		if(input<1 || node<1 || wmin > wmax){
			cout<< "output init fail : " << endl
				<< "node = " << node << endl
				<< "input = " << input << endl
				<< "wmin,wmax = " << wmin << ',' << wmax << endl;
			_init = false;
		}
		else{
			this->act = act;
			this->dact = dact;
			_node = node;
			_input = input;

			_offset = offset;
			_start = start;
			_outputs = outputs;

			desireOut = zeros<rowvec>(_node);

			weight = randu(_input, _node)*(wmax-wmin);
			mat mmin(_input, _node);
			mmin.fill(wmin);
			weight += mmin;

			sum = zeros<rowvec>(_node);
			out = zeros<rowvec>(_node);

			delta = zeros<rowvec>(_node);
			wupdate = zeros<mat>(_input, _node);
			wupdates = zeros<mat>(_input, _node);

			cost = zeros<rowvec>(_node);
			costnmse = zeros<rowvec>(_node);

			_init = true;
		}
	}


	void nnLayerOutput::setOutput(int Nth){
		int offsetN = _offset*Nth;
		//cout << "get O " << offsetN << " to " << offsetN + _node-1 << endl;
		for(int i=0; i<_node; ++i){
			//cout << "out get:"<<offsetN+i+_start<<endl;
			desireOut(i) = (*_outputs)( offsetN + i + _start);
		}
	}

}
