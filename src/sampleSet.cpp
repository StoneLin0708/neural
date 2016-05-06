#include "sampleSet.hpp"
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <time.h>

using std::swap;
using std::endl;
using std::cout;

sampleSet::sampleSet(type Type, param Param){
	_type = Type;
	_param.numberOfSample = Param.numberOfSample;
	_param.rangeStart = Param.rangeStart-1;
	_param.rangeEnd = Param.rangeEnd-1;
	_param.numberPerSet = Param.numberPerSet;

	srand(time(NULL));

	if(_type == sequence || _type == sequenceRange){
		counter_sample = _param.numberOfSample;
		_n_set=1;
	}else{
		_n_set = _param.numberOfSample/_param.numberPerSet;
		if( _param.numberOfSample%_param.numberPerSet != 0)
			++_n_set;
		counter_sample = _param.numberPerSet;
	}

	counter_set = 0;
	counter_last = 0;
	_last = false;
	setMax = _n_set-1;
	lastSetSample = _param.numberOfSample - ( (_n_set-1) * _param.numberPerSet );
	_set = initSet(_n_set);
	sampleOffset = _set[counter_set] * _param.numberPerSet;
}

bool sampleSet::last(){
	if(_last){
		_last = false;
		return true;
	}
	return false;
}

vector<int> sampleSet::initSet(int size){
	vector<int> r;
	for(int i=0; i<size; ++i){
		r.push_back(i);
	}
	return r;
}

int sampleSet::getNext(){
/*
	std::cout << ","<<setMax << ','<<lastSetSample << ',' << sampleOffset <<','<< counter_sample <<',' << counter_last << endl;
	for(int i=0;i<_set.size();++i)
		cout << _set[i]<<',';
	cout<<endl;
*/
	--counter_sample;


	switch(_type){
		case sequence:
			if( counter_sample < 0 ){
				counter_sample = _param.numberOfSample-1;
			}
			else if( counter_sample == 0 ){
				_last = true;
			}
			return counter_sample;

		case sequenceRange:
			if( counter_sample < _param.rangeStart ){
				counter_sample = _param.rangeEnd;
			}
			else if( counter_sample == _param.rangeStart ){
				_last = true;
			}
			return counter_sample;

		case sequenceSet:
			if( counter_sample == 0)
					_last = true;
			else if( counter_sample < 0 ){
				++counter_set;
				if( counter_set == setMax ){  //runout all set
					counter_sample = lastSetSample;
					sampleOffset = counter_set * _param.numberPerSet;
				}else if(counter_set > setMax){
					counter_sample = _param.numberPerSet-1;
					counter_set = 0;
					sampleOffset = 0;
				}else{
					counter_sample = _param.numberPerSet-1;
					sampleOffset = counter_set * _param.numberPerSet;
				}
			}
			return sampleOffset + counter_sample;

		case sortRandomSet:

			if( counter_sample == 0)
				if(_set[counter_set] == setMax &&
						counter_last != 0){
					--counter_last;
					return rand()%_n_set;
				}else
					_last = true;
			else if( counter_sample < 0 ){
				++counter_set;
				if(counter_set > setMax){
					counter_set = 0;
					for(int i=0; i<setMax; ++i){
						swap(_set[rand()%_n_set],_set[rand()%_n_set]);
					}
					if(_set[0] == setMax)
						counter_sample = lastSetSample-1;
					else
						counter_sample = _param.numberPerSet-1;
					sampleOffset = _set[counter_set] * _param.numberPerSet;
				}else{
					if(_set[counter_set] == setMax){
						counter_sample = lastSetSample-1;
						counter_last = _param.numberPerSet-
							lastSetSample;
						sampleOffset = setMax * _param.numberPerSet;
					}else{
						counter_sample = _param.numberPerSet-1;
						sampleOffset = _set[counter_set] * _param.numberPerSet;
					}
				}
			}


			return sampleOffset + counter_sample;
		case fullRandomSet:
			if( counter_sample == 0)
					_last = true;
			if( counter_sample < 0 ){
				++counter_set;
				if( counter_set == setMax ){  //runout all set
					counter_sample = lastSetSample;
					sampleOffset = rand()%setMax * _param.numberPerSet;

				}else if(counter_set > setMax){
					counter_sample = _param.numberPerSet-1;
					counter_set = 0;
					sampleOffset = rand()%setMax * _param.numberPerSet;
				}else{
					counter_sample = _param.numberPerSet-1;
					sampleOffset = rand()%setMax * _param.numberPerSet;
				}
			}
			return sampleOffset + counter_sample;

	}

}

