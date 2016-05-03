#include "neural.hpp"

int nn::getFirstTestSample(){
	switch(_param.testType){
	case nn_t::testAll :
			return 0;
		break;
	case nn_t::testNumber :
			return _param.testStart;
		break;
	}
	return -1;
}

bool nn::getNextTestSample(int &sample){
	static int counter = _param.testStart+1;
	static nn_t::test_t testType = _param.testType;
	static int testStart	= _param.testStart;
	static int testEnd	= _param.testEnd;
	switch(testType){
	case nn_t::testAll :
		sample = counter;
		if( counter ==  _n_sample){
			counter = 1;
			sample = 0;
			return false;
		}
		++counter;
		break;
	case nn_t::testNumber :
		sample = counter;
		if( counter == testEnd){
			counter = testStart;
			return false;
		}
		++counter;
		break;
	}
	return true;
}

int nn::getFirstSample(){
	switch(_param.trainType){
	case nn_t::trainAll :
			return 0;
		break;
	case nn_t::trainNumber :
			return _param.trainStart-1;
		break;
	case nn_t::trainBunch :
			return 0;
		break;
	}
	return -1;
}

bool nn::getNextSample(int &sample){
	static int counter = _param.trainStart;
	static nn_t::train_t trainType = _param.trainType;
	static int trainStart = _param.trainStart;
	static int trainEnd = _param.trainEnd;
	static int trainNumber = _param.trainNumber;
	static int bunch_counter = 1;

	switch(trainType){
	case nn_t::trainAll :
		sample = counter;
		if( counter ==  _n_sample){
			counter = 1;
			sample = 0;
			return false;
		}
		++counter;
		break; case nn_t::trainNumber :
		sample = counter;
		if( counter == trainEnd){
			counter = trainStart-1;
			sample = counter;
			++counter;
			return false;
		}
		++counter;
		break;
	case nn_t::trainBunch :
		sample = counter;

		if( counter == _n_sample || counter == trainEnd){
			counter = 1;
			bunch_counter = 1;
			sample = 0;
			return false;
		}
		if(	counter % trainNumber == 0){
			counter = bunch_counter * trainNumber+1;
			sample = counter-1;
			++bunch_counter;
			return false;
		}
		++counter;
		break;
	}
	return true;
}
