#include "sample.hpp"
#include "stringCheck.hpp"
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace std;

sample::sample(){
	_nlabel = 0;
	_nfeature = 0;
}

bool sample::readFormat(const string& in, data_t& out){
	out.label.clear();
	out.feature.clear();
	vector<string> lf = split(in,':');
	vector<string> l;
	if(!nolable){

	if( lf.size() != 2){
		errorString(" sample format error ",in,
		" ex. label0,label1,...,labelN:feature0,feature1,..., featureN"
		);
		return false;
	}
	l = split(lf[0],',');
	if( l.size() == 0){
		errorString(" sample error ", in," no label");
		return false;
	}
	else if( _nlabel == 0)
		_nlabel = l.size();
	else if( l.size() != _nlabel ){
		errorString(" sample error ", in," label number not equal");
		return false;
	}

	}
	vector<string> f = split(lf[1],',');
	if( f.size() == 0){
		errorString(" sample error ", in," no feature");
		return false;
	}
	else if( _nfeature == 0){
		_nfeature = f.size();
	}
	else if( f.size() != _nfeature ){
		errorString(" sample error ", in," feature number not equal");
		return false;
	}

	for(int i = 0; i<_nlabel; ++i){
		if( !isDouble( l[i] ))
			return false;
		out.label.push_back( atof(l[i].c_str()) );
	}

	for(int i = 0; i<_nfeature; ++i){
		if( !isDouble( f[i] ))
			return false;
		out.feature.push_back( atof(f[i].c_str()) );
	}

	return true;
}

bool sample::read(const string path,bool nolable){
	this->nolable = nolable;
    ifstream sample_f;
    sample_f.open(path.c_str(), ios::in);
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
	cout << " output : " << _nlabel;
	cout << " feature : " << _nfeature;
	cout << " sample : " << _data.size() << endl;
    sample_f.close();
	_path = path;
    return true;
}

void sample::clear(){
	_data.clear();
}

void sample::list(){
	cout<< " label : " << _nlabel
		<< " feature : " << _nfeature
		<< " sample : " << size() << endl;

	for(int i=0; i<(int)_data.size(); i++){
		cout<< "[ ";
		for(int j=0; j< (int)_nlabel; j++)
			cout<< setw(5) << _data[i].label[j] << " ,";
		for(int j=0; j<(int)_nfeature; j++)
			cout<< setw(5) << _data[i].feature[j] << " ,";
		cout<< " ]"  << endl;
	}
}

