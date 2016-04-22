#pragma once
#include <vector>
#include <string>

/*
typedef struct data_t{
	int l;
	float x;
	float y;
}data_t;
*/

class data_t{
public:
	int l;
	std::vector<double> feature;

};

typedef std::vector<data_t> data_v;

class sample{
public:
	sample();

	bool read(const char* path);
	bool save();
	bool saveTo(const char* path);
	data_t& operator[](int i);
	unsigned int size();
	void clear();

	void list();

	bool isInt(std::string& test);
	bool isFloat(std::string& test);

private:
	data_v _data;
	std::string _path;
	void errString(std::string& line, std::string& str,int s ,int e);
	bool readFeature(std::string& in, double& out,int& s,int& e);
	bool readFormat(std::string& in, data_t& out);

};

