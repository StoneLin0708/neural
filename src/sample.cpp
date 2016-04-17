#include "sample.hpp"
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

bool sample::isInt(std::string& test){
	istringstream iss(test);
	int f;
	iss >> noskipws>> f;
	return iss.eof() && !iss.fail();
}

bool sample::isFloat(std::string& test){
	istringstream iss(test);
	float f;
	iss >> noskipws>> f;
	return iss.eof() && !iss.fail();
}

void sample::errString(std::string& line, std::string& str,
						 int s ,int e){
	cout << " sample error in line  \"" << line << "\""<< endl
		 << "  at character " << s << " to " << e
		 << " \""<< str << "\""<< endl;
}

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
	if( !isFloat(tmp) ){
		errString(in,tmp,s,e);
		return false;
	}
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

bool sample::read(const char* path){
    ifstream sample_f;
    sample_f.open(path, ios::in);
	string in;
	data_t tmp;

    if ( sample_f.fail() ){
        printf("fail to open file : %s\n", path);
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

void sample::list(){
	cout << "[";
	for(int i=0; i<(int)_data[0].feature.size(); i++)
		cout << setw(4) << i << " ,";
	cout << "    l ]" << endl;

	for(int i=0; i<(int)_data.size(); i++){
		cout << "[ ";
		for(int j=0; j<(int)_data[0].feature.size(); j++)
			cout << setw(5) << _data[i].feature[j] << " ,";
		cout << setw(3) << _data[i].l << " ]"  << endl;
	}
}

