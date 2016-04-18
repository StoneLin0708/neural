#include "neural.hpp"
#include <iostream>

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
	learning_rate = dlearning_rate;
}

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
	mat od = zeros(output_num);
	for(int i=0; i<output_num; ++i){
		od.at(i) = (de[i] - oo[i]) * dactivation(os[i]);
		for(int j=0; j<hidden_num+1; ++j){
			odel.at(j,i) =
				learning_rate * od.at(i) * ho[j];
			//cout << i << ',' << j << ':' <<odel.at(j,i) << endl;
		}
	}
	//output = output + odel;
	//hidden to input
	mat hd = zeros(hidden_num+1);
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

void nn::clear_dels(){
	hdels = zeros(input_num+1, hidden_num+1); //hidden delta
	odels = zeros(hidden_num+1, output_num); //output delta
}

void nn::train(){
	test();
	cal_del();
}

