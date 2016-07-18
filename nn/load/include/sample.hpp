#pragma once
#include <vector>
#include <string>
#include <armadillo>

using std::string;

namespace  nn{

class Sample{
public:
    Sample();

    bool read(const string path);

    arma::mat input;
    arma::mat output;

	void list();

    int n_sample;
    int n_input;
    int n_output;


};

}
