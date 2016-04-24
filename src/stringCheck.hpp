#pragma once
#include <string>
#include <vector>

using std::string;
using std::vector;

bool isInt(const string &testString, bool errmsg=true);
bool isFloat(const string &testString, bool errmsg=true);
bool readFor(const string &text, const string &in, string &out);
void errorString(const string &msg, const string &error,
		const string &right);
vector<string> &split(const string &s, char delim,
		vector<string> &elems);
vector<string> split(const string &s, char delim);

