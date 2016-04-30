#pragma once
#include "neural.hpp"
#include <string>

using std::string;

double drawResult(nn& n,string title);
double drawError(nn& n, int iteration, string title);
