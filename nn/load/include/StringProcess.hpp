#pragma once
#include <string>
#include <vector>

using std::string;
using std::vector;

void replaceChar(string &str, char find, char replace);
void removeChar(string &str, char remove);
void removeCharFront(string &str, char remove);

void removeChar(string &str, vector<char> &ch);
bool isInt(const string &testString, bool errmsg=true);
bool isFloat(const string &testString, bool errmsg=true);
bool isDouble(const string &testString, bool errmsg=true);
bool readFor(const string &text, const string &in, string &out);
void errorString(const string &msg, const string &error, const string &right);

vector<string> &split(const string &s, char delim, vector<string> &elems);
vector<string> split(const string &s, char delim);

