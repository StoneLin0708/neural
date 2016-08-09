#pragma once
#include "model/ANNModel.hpp"
#include <string>

using std::string;

bool drawResult2D(nn::ANNModel &, bool show=true);
/*
double drawResultTimeseries(nn& n,string title, string name);
double drawResult(nn& n,string title,string name);
double drawError(nn& n, int iteration, string name);
*/
