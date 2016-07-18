#pragma once
#include <armadillo>
#include <vector>
#include "load/include/sample.hpp"

using std::vector;
using arma::rowvec;

namespace  nn{

    class sampleFeeder{
    public:
        sampleFeeder(Sample *, rowvec *in, rowvec *out);

        virtual void reset();
        virtual void next();
        virtual bool isLast();

    protected:
        int iter;
        int n_sample;
        int n_input;
        int n_output;
        Sample *s;
        rowvec *in;
        rowvec *out;

    };
/*
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
*/
}
