#include <sstream>
#include <iostream>
#include <iomanip>
#include "../include/stringProcess.hpp"

using std::cout;
using std::endl;
using std::string;
using std::istringstream;
using std::stringstream;
using std::noskipws;


void replaceChar(string &str, char find, char replace){
	for(int i=0;i<str.size();++i){
		if(str[i] == find)
			str[i] = replace;
	}
}

void removeCharFront(string &str, char remove){
	string tmp;
	bool rm = true;
	for(int i=0;i<str.size();++i){
		if( !rm || (str[i] != remove) ){
			rm=false;
			tmp.push_back(str[i]);
		}
	}
	str = tmp;
}

void removeChar(string &str, char remove){
	string tmp;
	for(int i=0;i<str.size();++i){
		if(str[i] != remove)
			tmp.push_back(str[i]);
	}
	str = tmp;
}

void removeChar(string &str, vector<char> &ch){
	string tmp;
	bool accept;
	for(int i=0;i<str.size();++i){
		accept = true;
		for(int j=0;j<ch.size();++j){
			if(str[i] == ch[j])
				accept = false;
		}
		if(accept) tmp.push_back(str[i]);
	}
	str = tmp;
}

bool isInt(const string &testString, bool errmsg){
	istringstream iss(testString);
	int f;
	iss >> noskipws>> f;
	bool r = iss.eof() && !iss.fail();
	if(r) return true;
	if(errmsg)
		errorString(" error data type \"",testString,"\" not a int");
	return false;
}

bool isFloat(const string &testString, bool errmsg){
	istringstream iss(testString);
	float f;
	iss >> noskipws>> f;
	bool r = iss.eof() && !iss.fail();
	if(r) return true;
	if(errmsg)
		errorString(" error data type \"",testString,"\" not a float");
	return false;
}

bool isDouble(const string &testString, bool errmsg){
	istringstream iss(testString);
	double f;
	iss >> noskipws>> f;
	bool r = iss.eof() && !iss.fail();
	if(r) return true;
	if(errmsg)
		errorString(" error data type \"",testString,"\" not a double");
	return false;
}

bool readFor(const string &text, const string &in, string& out){
	int i=0;
	string tmp;
	out.clear();
	while(in[i] == text[i]){
		//cout<< in[i] << " : "<< text[i] << endl;
		if((int)text.size() == i+1){
			for(int j=in.size()-1; j>=(int)text.size(); --j){
				tmp.push_back(in[j]);
			}
			out.clear();
			for(int j=tmp.size()-1; j>=0; --j){
				out.push_back(tmp[j]);
			}
			//cout<<" read " << text << " : "<< out << endl;
			return true;
		}
		i++;
	}
	return false;
}

void errorString(const string &msg, const string &error,
		const string &right){
	cout<< msg << " : " << error;
	if(right.size() != 0)
		cout<< " : "<< right;
	cout<<endl;
}

vector<string> &split(const string &s, char delim,
		vector<string> &elems){
	stringstream ss(s);
	string item;
	while(std::getline(ss, item, delim)){
		if (!item.empty())
			elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim){
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

