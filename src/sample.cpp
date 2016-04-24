#include "sample.hpp"
#include "stringCheck.hpp"
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace std;

sample::sample(){

}

data_t& sample::operator[](int i){
	return _data[i];
};

unsigned int sample::size(){
	return _data.size();
};

bool sample::readFeature(std::string& in, double& out,
		int& s,int& e){
	string tmp;

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
	if( !isFloat(tmp) )
		return false;

	out = (double)atof( tmp.c_str() );
	return true;
}

bool sample::readFormat(std::string& in,data_t& out){
	int s,e,f;
	string stmp;
	double dtmp;

	out.feature.clear();
	s = 0;
	e = in.find(':', s);
	for(int i=0; i<e; ++i){
		stmp.push_back( in[i]);
	}
	if( !isInt(stmp) )
		return false;

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

bool sample::read(const char* path){
    ifstream sample_f;
    sample_f.open(path, ios::in);
	string in;
	data_t tmp;

    if ( sample_f.fail() ){
        cout<< "fail to open file : " << path<< endl;
        return false;
    }

    while( sample_f >> in ){
		if( readFormat(in,tmp) )
			_data.push_back(tmp);
		else
			return false;
    }

    sample_f.close();
	_path = path;
    return true;
}

void sample::clear(){
	_data.clear();
}

void sample::list(){
	cout << "[  ";
	for(int i=0; i<(int)_data[0].feature.size(); i++)
		cout<< setw(4) << i << " ,";
	cout<< "    l ]" << endl;

	for(int i=0; i<(int)_data.size(); i++){
		cout<< "[ ";
		for(int j=0; j<(int)_data[0].feature.size(); j++)
			cout<< setw(5) << _data[i].feature[j] << " ,";
		cout<< setw(3) << _data[i].l << " ]"  << endl;
	}
}

