#pragma once
#include "neural.hpp"
#include <string>

using std::string;

double drawResultTimeseries(nn& n,string title, string name);
double drawResult(nn& n,string title,string name);
double drawError(nn& n, int iteration, string name);
