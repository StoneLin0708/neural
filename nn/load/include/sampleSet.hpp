#pragma once
#include <vector>

using std::vector;

class sampleSet{
public:
	typedef enum{
		sequence,
		sequenceRange,
		sequenceSet,
		sortRandomSet,
		fullRandomSet
	}type;

	typedef struct param{
		int numberOfSample;
		int rangeStart;
		int rangeEnd;
		int numberPerSet;
	}param;

	sampleSet(type Type, param Param);

	int getNext();
	bool last();
	//reset();
private:
	bool _last;
	type _type;
	param _param;

	int _n_set;

	int counter_set;
	int counter_sample;


	vector<int> initSet(int size);
	int setMax;
	int lastSetSample;
	int counter_last;
	vector<int> _set;
	int sampleOffset;
};
