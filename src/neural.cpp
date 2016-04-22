#include "neural.hpp"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
/*
nlayer::nlayer(){
	_init = false;
}
*/
nlayer::nlayer(
		arma::rowvec& input, layer_t type, int n_nodes,
		double (*activation)(double in),
		double (*activation)(double in) ){
	if( input.size() > 1 &&
		n_nodes < node_max &&
		n_nodes > 1 &&
		activation != NULL &&
		dactivation != NULL ){
		_initialize(input,type,n_nodes,activation,dactivation);
		_init = true;
	}
	else{
		cout<< "fail initialize layer with parameter : " << endl
			<< " input   : " << endl << input << endl;
		if(type)
		cout<< " type    : " << hidden << endl;
		else
		cout<< " type    : " << output << endl;

		cout<< " n_nodes : " << n_nodes << endl
			<< " act     : " << activation << endl
			<< " dact    : " << dactivation << endl;
	}
}

void nlayer::_initialize(
		arma::rowvec& input, layer_t type, int n_nodes,
		double (*activation)(double in),
		double (*activation)(double in) ){
	_input = input.n_cols;
	_nodes = n_nodes + type;
	i = &input;
	random_w(-0.5, 0.5);
	s = zeros(_nodes);
	o = zeros(_nodes);
	d = zeros(_input, _nodes);
	ds = zeros(_input, _nodes);
}

void nlayer::random_w(double min, double max){
	w = randu(_input, _nodes)*(max-min);
	mat mmin(_input, _nodes);
	mmin.fill(min);
	w += mmin;
}

void nlayer::show(){

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
	learning_rate = dlearning_rate;
	normalize_scale = 1.0;
}
*/
nn::nn(string& path){
}

void nn::randomInit(){
	hidden = randu<mat>(input_num+1, hidden_num+1);
	output = randu<mat>(hidden_num+1, output_num);
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

bool nn::readFor(int line, string& in, const string text){
	cout<< "nn read fail at " << line
		<< " : "<< in << endl;
	return false;
}

void nn::errString(std::string& line, std::string& str,
		int s ,int e){
	cout<< " neural error in line  \"" << line << "\""<< endl
		<< "  at character " << s << " to " << e
		<< " \""<< str << "\""<< endl;
}

bool nn::readLayer(int line, std::string& in){
	int s,e;
	string tmp;
	nlayer::layer_t type;
	if(readFor(line, in, "hidden=")){
		type = nlayer::hidden;
		s = 0;
		e = in.find(',',s);
		if(e == string::npos){
			errString(in, tmp, s, in.end());
			return false;
		}
		else{
			for(int i=s; i<e; ++i)
				tmp.push_back(in[i]);
		}
	}
	else if(readFor(line, in, "output=")){
		type = nlayer::hidden;
	}
	else
		return false;

	e = in.find(',',s);
	if(e == string::npos){
		e = in.find(')',s);
		if(e == string::npos){
			return false;
		}
	}
	for(int i=s; i<e; i++){
		tmp.push_back( in[i]);
	}
	//cout << "in " << in << " s= " << s << " e= " << e << " : " << tmp << endl;
	if( !isFloat(tmp) ){
		errString(in,tmp,s,e);
		return false;
	}
	out = (double)atof( tmp.c_str() );
	return true;
}

bool nn::readnn(std::string& path){
    ifstream sample_f;
    sample_f.open(path, ios::in);
	string in;

    if ( sample_f.fail() ){
        cout<< "fail to open file : " << path<< endl;
        return false;
    }
	int line=0;
    while( sample_f >> in ){
		line++;
		switch(line){
		case 1:
			if( readFor(line, in, "sample data=") ){
				if(!readSample(in))
					return false;
			}
			else
				return false;
			break;
		case 2:
			if( readFor(line, in, "sample scale=") ){
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
			if( readFor(line, in, "learning rate=") ){
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
			if( readFor(line, in, "input feature=") ){
				if(_s.isInt(in))
					input_num = atoi(in.c_str());
				else{
					cout<<" data wrong ! "<< line<< " : "<< in<< endl;
					return false;
				}
			}
			else
				return false;
			break;
		defult :
			if( !readLayer(line, in) )
				return false;
			break;
		}
    }

    sample_f.close();

	int s,e,f;
	string stmp;
	double dtmp;

	out.feature.clear();
	s = 0;
	e = in.find(':', s);
	for(int i=0; i<e; ++i){
		stmp.push_back( in[i]);
	}
	if( !isInt(stmp) ){
		errString(in,stmp,s,e);
		return false;
	}
	out.l = atoi( stmp.c_str() );
	stmp.clear();
	f = 0;
	s = e+2;
	while(readFeature(in, dtmp, s, e)){
		s = e+1;
		out.feature.push_back(dtmp);
		++f;
	}

	return true;
}

void nn::showw(){
	//cout << "hidden" << endl << hidden << endl;
	//cout << "output" << endl << output << endl;

}

void nn::showdw(){
	/*
	cout << "hdel" << endl << hdel << endl;
	cout << "odel" << endl << odel << endl;
	*/
}

void nn::showd(){
	/*
	cout << "input.t" << endl << input.t() << endl;
	cout << "hs" << endl << hs << endl;
	cout << "ho.t" << endl << ho.t() << endl;
	cout << "os" << endl << os << endl;
	cout << "oo.t" << endl << oo.t() << endl;
	cout << "de.t" << endl << de.t() << endl;
	*/
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

void nn::clear_dels(){
	for(int i=0; i<layer.size(); ++i)
		layer[i].dels.zeros();
}

void nn::cal_del(){
	/*
	//output to hidden
	for(int i=0; i<output_num; ++i){
		od(i) = (de(i) - oo(i)) * dactivation(os(i));
		for(int j=0; j<hidden_num+1; ++j){
			odel(j,i) =
				learning_rate * od(i) * ho(j);
		}
	}
	//hidden to input
	for(int layer;
	hd.zeros();
	for(int i=0; i<hidden_num+1; ++i){
		for(int k=0; k<output_num; ++k){
			hd(i) += od(k)*output(i,k);
		}
		hd(i) *= dactivation(hs(i));
		for(int j=0; j<input_num+1; ++j)
		{
			hdel(j,i) =
				learning_rate * hd(i) * input(j);
		}
	}
	*/
	//output to hidden
	nlayer& o = layer[layer.size()];
	for(int i=0; i<o.n_nodes; ++i){
		o.d(i) = (de(i) - o.o(i)) * o.dact(o.s(i));
		for(int j=0; j<layer[layer.size()-1].n_nodes(); ++j){
			odel(j,i) =
				learning_rate * o.d(i) * layer[layer.size()-1].o(j);
		}
		o.dels += o.del;
	}
	//hidden to input
	int nodes, nodes_;
	for(int l = layer.size()-1; l>0; ++l){
		nlayer& nl = layer[l];
		nlayer& nl_ = layer[l];
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
					learning_rate * nl.d(i) * *(nl.i)(j);
			}
		}
		nl.dels += nl.del;
	}

}

void nn::wupdate(){
	for(int i=0; i<layer.size(); ++i)
	{
		layer[i].w += layer[i].dels;
	}
}

void nn::showsd(){
	/*
	double sod=0, shd=0;
	for(int i=0; i<output_num; ++i)
		sod += abs(ods(i))*1000;
	for(int i=0; i<hidden_num+1; ++i)
		shd += abs(hds(i))*1000;
	cout<< "error sum at output = " << setw(8) << sod
		<< " at hidden = " << setw(8) << shd << endl;
	e.push_back(sod);
	*/
}


void nn::train(int iteration){
	int fnum = (int)_s[0].feature.size();

	for(int i=0; i<iteration; ++i){
		clear_dels();
		for(int s=0; s<(int)_s.size(); ++s){
			//set input
			//get sample performance is poor,need rework
			for(int j=0; j<fnum; ++k)
				input(j) = _s[s].feature[j]*normalize_scale;
			//set desire output
			de.fill(0);
			de.at( _s[s].l ) = 1;
			//forward
			test();
			//back propagation
			cal_del();
		}
/*
		showw();
		showd();
		showdw();
*/
		/*
		if(j%50 == 0){
		//if(false){
			cout << "i = " << setw(5) << i << ' ';
			showsd();
		}
		*/
		//finish a iteration, change neural weight
		wupdate();
	}
/*
	showd();
	showw();
	showdw();
*/
}

