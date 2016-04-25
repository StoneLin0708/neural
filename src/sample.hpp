#pragma once
#include <vector>
#include <string>

using std::vector;
using std::string;

class data_t{
public:
	vector<double> label;
	vector<double> feature;
};

typedef std::vector<data_t> data_v;

class sample{
public:
	sample();

	bool read(const string path);
	bool save();
	bool saveTo(const char* path);

	data_t& operator[](int i){return _data[i];};
	size_t size(){return _data.size();};

	void clear();

	void list();

	size_t n_label(){return _nlabel;};
	size_t n_feature(){return _nfeature;};

private:
	size_t _nlabel;
	size_t _nfeature;

	data_v _data;
	std::string _path;
	bool readFormat(const string& in, data_t& out);

};

