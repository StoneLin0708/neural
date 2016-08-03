#include "load/include/Sample.hpp"
#include "load/include/StringProcess.hpp"

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace std;

using std::vector;

namespace nn {

class data_t{
public:
    vector<double> input;
    vector<double> output;
};

typedef std::vector<data_t> data_v;

Sample::Sample(){
}

static bool readFormat(const string& in, data_t& out){
    out.input.clear();
    out.output.clear();
    vector<string> sp = split(in,':');
	vector<string> l;

    if( sp.size() == 1){
        auto sInput = split(sp[0],',');
        for(int i=0; i<(int)sInput.size(); ++i){
            if( !isDouble(sInput[i]) ) return false;
            out.input.push_back( atof( sInput[i].c_str() ) );
        }

    }
    else if( sp.size() == 2){
        auto sOutput = split(sp[0],',');
        auto sInput = split(sp[1],',');
        for(int i=0; i<(int)sInput.size(); ++i){
            if( !isDouble(sInput[i]) ) return false;
            out.input.push_back( atof( sInput[i].c_str() ) );
        }
        for(int i=0; i<(int)sOutput.size(); ++i){
            if( !isDouble(sOutput[i]) ) return false;
            out.output.push_back( atof( sOutput[i].c_str() ) );
        }
    }
    else
        return false;

    return true;
}


bool Sample::read(const string path){
    ifstream sample_f;
    sample_f.open(path.c_str(), ios::in);
	string in;
	data_t tmp;
    data_v data;
    int line = 0;

    if ( sample_f.fail() ){
        cout<< "fail to open file : " << path<< endl;
        return false;
    }

    while( sample_f >> in ){
        line++;
        replaceChar(in, '\t', ' ');
        removeChar(in, ' ');
        if( readFormat(in, tmp) )
            data.push_back(tmp);
        else{
            cout<< "sample read fail at line " << line << " : " << in;
			return false;
        }
    }
    sample_f.close();

    n_sample = data.size();
    n_input = data[0].input.size();
    n_output = data[0].output.size();

    input.zeros(data.size(), n_input);
    output.zeros(data.size(), n_output);

    for(int i=0; i< n_sample; ++i){
       if((int)data[i].input.size() != n_input ||
          (int)data[i].output.size() != n_output){
           cout<< "sample error at "<< i <<endl;
           return false;
       }
    }

    for(int i=0; i< n_sample; ++i){
        for(int j=0; j<n_input; ++j)
            input(i,j) = data[i].input[j];
        for(int j=0; j<n_output; ++j)
            output(i,j) = data[i].output[j];
    }

    return true;
}

void Sample::list(){

    cout<< " input : " << n_input
        << " output : " << n_output
        << " size : " << n_sample << endl;

    for(int i=0; i< n_sample; i++){
        cout<< " in:" <<input(i)
            << " out:" << output(i) <<endl;
	}
}

}
