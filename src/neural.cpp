#include "neural.hpp"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;

nn::nn(){
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
	ods = zeros(output_num);
	hds = zeros(hidden_num+1);
	learning_rate = dlearning_rate;
}

sample& nn::getSample(){
	return _s;
}

bool nn::readSample(std::string& path){
	if( _s.read(path.c_str()) ){
		_s.list();
		return true;
	}
	else
		return false;
};

void nn::randomInit(){
	hidden = randu<mat>(input_num+1, hidden_num+1);
	output = randu<mat>(hidden_num+1, output_num);
}

void nn::showw(){
	cout << "hidden" << endl << hidden << endl;
	cout << "output" << endl << output << endl;

}

void nn::showdw(){
	cout << "hdel" << endl << hdel << endl;
	cout << "odel" << endl << odel << endl;

}

void nn::showd(){
	cout << "input.t" << endl << input.t() << endl;
	cout << "hs" << endl << hs << endl;
	cout << "ho.t" << endl << ho.t() << endl;
	cout << "os" << endl << os << endl;
	cout << "oo.t" << endl << oo.t() << endl;
	cout << "de.t" << endl << de.t() << endl;
}

void nn::test(){
	//forward
	hs = input.t() * hidden;
	for(int i=0; i<hidden_num; ++i)
		ho.at(i) = activation( hs.at(i) );
	os = ho.t() * output;
	for(int i=0; i<output_num; ++i)
		oo.at(i) = activation( os.at(i) );

}

void nn::cal_del(){
	//output to hidden
	for(int i=0; i<output_num; ++i){
		od.at(i) = (de[i] - oo[i]) * dactivation(os[i]);
		for(int j=0; j<hidden_num+1; ++j){
			odel.at(j,i) =
				learning_rate * od.at(i) * ho[j];
			//cout << i << ',' << j << ':' <<odel.at(j,i) << endl;
		}
	}
	//hidden to input
	for(int i=0; i<hidden_num+1; ++i){
		for(int k=0; k<output_num; ++k){
			hd.at(i) += od.at(k)*output.at(i,k);
		}
		hd.at(i) *= dactivation(hs[i]);
		for(int j=0; j<input_num+1; ++j)
		{
			hdel.at(j,i) =
				learning_rate * hd.at(i) * input[j];
		}
	}

}


void nn::wupdate(){
	hidden += hdels;
	output += odels;
}

void nn::showsd(){
	double sod=0, shd=0;
	for(int i=0; i<output_num; ++i)
		sod += abs(ods[i])*1000;
	for(int i=0; i<hidden_num+1; ++i)
		shd += abs(hds[i])*1000;
	cout<< "error sum at output = " << setw(8) << sod
		<< " at hidden = " << setw(8) << shd << endl;
}

void nn::clear_dels(){
	hdels = zeros(input_num+1, hidden_num+1); //hidden delta
	odels = zeros(hidden_num+1, output_num); //output delta
	ods = zeros(output_num);
	hds = zeros(hidden_num+1);
}

void nn::train(int iteration){
	for(int j=0; j<iteration; ++j){
		clear_dels();
		for(int i=0; i<_s.size(); i++){
			input.at(0) = _s[i].feature[0]/10;
			input.at(1) = _s[i].feature[1]/10;
			de.fill(0);
			de.at( _s[i].l ) = 1;
			//n.train();
			test();
			cal_del();
			ods += od;
			hds += hd;
			odels += odel;
			hdels += hdel;
			//cout << "1-o = " << 1-n.oo[s[i].l] << endl;
		}
		cout << "i = " << setw(5) << j << ' ';
/*
		showw();
		showd();
		showdw();
*/
		showsd();
		wupdate();
	}
	showw();
/*
	showd();
	showdw();
*/
}

