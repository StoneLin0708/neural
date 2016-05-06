#include "../src/stringCheck.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;
void removeSpace(string &str){
	string tmp;
	for(int i=0;i<str.size();++i)
		if(str[i] != ' ' && str[i] != 13)
			tmp.push_back(str[i]);
	str = tmp;
}

int main(int argc, char* argv[]){
	if(argc != 3){
		cout << " in, out " << endl;
		return -1;
	}
	ifstream is;
	is.open(argv[1], ios::in);

	ofstream os;
	os.open(argv[2], ios::out);

	int i;
	string in;
	string out;

	if( !is.is_open() ){
		cout << " fail open : " << argv[0] << endl;
	}
	if( !os.is_open() ){
		cout << " fail open : " << argv[1] << endl;
	}

	is >> ws;
	while( getline(is,in) ){
		vector<string> ss = split(in,',');

		for(int j=0;j<ss.size();++j)
			removeSpace(ss[j]);

		out = ss.back();
		out.push_back(':');
		for(i=0; i<ss.size()-2; ++i){
			out += ss[i];
			out.push_back(',');
		}
		out += ss[i];
		//cout << "ooooooooo " << out << endl;
		os << out << endl;
		out.clear();
	}

	is.close();
	os.close();
}
