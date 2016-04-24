#include <sstream>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

vector<string> &split(const string &s, char delim,
		vector<string> &elems){
	stringstream ss(s);
	string item;
	while(std::getline(ss, item, delim)){
		elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim){
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

int main(){
	string in = "aadfasdfa=,1,3,4,5";
	vector<string> out = split(in,',');
	for(int i=0; i<out.size(); ++i)
		cout << i << " :\"" << out[i] << "\"" << endl;
	return 0;
}
