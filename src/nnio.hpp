#include "neural.hpp"
#include <string>
using std::string;
bool readlayerParam(const string &in,layerParam &lparam);
int readnnWhich(const string &in, string &out);
bool readnn(const string& path, nnParam &param);
