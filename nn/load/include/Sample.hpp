#pragma once
#include <vector>
#include <string>
#include <armadillo>
#include "method/include/Normailze.hpp"

using std::string;

namespace  nn{

class Sample{
public:
    Sample();

    bool read(const string path);

    arma::mat input;
    arma::mat output;

    std::vector<double> outputMap;
    NormParam norm_in;
    NormParam norm_out;

    void list();

    int n_sample;
    int n_input;
    int n_output;


};

}
