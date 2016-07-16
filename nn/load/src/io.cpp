#include <stringProcess.hpp>
#include "sampleSet.hpp"

#include <fstream>
#include <vector>
#include <float.h>

#include "nnio.hpp"

using namespace std;


bool readnnSampleSet(const string &in,sampleSet::type &type, int &start, int &end, int &numberOfSet){
	vector<string> sp = split(in,',');
	if( sp[0] == "sequence" ){
		type = sampleSet::sequence;
		return true;
	}
	else if( sp[0] == "sequenceRange" ){
		type = sampleSet::sequenceRange;
		if(sp.size() != 3) return false;
		if( !isInt(sp[1]) || !isInt(sp[2]) ) return false;
		start = atoi(sp[1].c_str());
		end = atoi(sp[2].c_str());
		return true;
	}
	else if( sp[0] == "sequenceSet" ){
		type = sampleSet::sequenceSet;
		if(sp.size() != 2) return false;
		if(!isInt(sp[1])) return false;
		numberOfSet = atoi( sp[1].c_str() );
		return true;
	}
	else if( sp[0] == "sortRandomSet" ){
		type = sampleSet::sortRandomSet;
		if(sp.size() != 2) return false;
		if(!isInt(sp[1])) return false;
		numberOfSet = atoi( sp[1].c_str() );
		return true;
	}
	else if( sp[0] == "fullRandomSet" ){
		type = sampleSet::fullRandomSet;
		if(sp.size() != 2) return false;
		if(!isInt(sp[1])) return false;
		numberOfSet = atoi( sp[1].c_str() );
		return true;
	}

	errorString(" readnn fail sampling error ", in,"");
	return false;
}

bool readLayerParam_hidden(const string &in,layerParam &lparam){
	vector<string> sp = split(in,',');
	if(sp.size() != 3 && sp.size() != 2) return false;
	if( !isInt(sp[0]) ) return false;
	lparam.level = atoi(sp[0].c_str());

	if( !isInt(sp[1]) ) return false;
	lparam.nodes = atoi(sp[1].c_str());
	if(sp.size() == 3) lparam.activation = sp[2];
	return true;
}

bool readLayerParam_output(const string &in,layerParam &lparam){
	vector<string> sp = split(in,',');
	if(sp.size() != 1 && sp.size() != 2) return false;
	lparam.level = -1;

	if( !isInt(sp[0]) ) return false;
	lparam.nodes = atoi(sp[0].c_str());
	if(sp.size() == 2) lparam.activation = sp[1];
	return true;
}


bool readnn(const string& path, nnParam &param){
	ifstream fnn;
	fnn.open(path.c_str(), ios::in);
	string in,out;
	layerParam lparam;

	param.sampleType = nn_t::empty;		/*0*/
	param.stopTrainingCost = -DBL_MAX;	/*1*/
	param.trainFeature = 0;				/*2*/
	param.sampleData = "";				/*n 3*/

	param.iteration = 0;				/*n 4*/
	param.learningRate = 0;				/*n 5*/
	//param.hidden;						/*6*/
	//param.output;						/*n 7*/

	param.normalizeMethod = "";			/*8*/
	param.loadWeight = false;			/*9*/
	param.saveWeight = false;			/*10*/
	param.weightPath = "";				/*11*/

	param.weightName = "";				/*12*/
	param.defaultActivation = "";		/*n 13*/
	param.featureOffset = 0;			/*14*/

	param.trainType = sampleSet::sequence;	/*15*/
	param.trainStart = 0;				/*15*/
	param.trainEnd = 0;					/*15*/
	param.trainNumber = 0;				/*15*/

	param.testType = sampleSet::sequence;		/*16*/
	param.testStart = 0;				/*16*/
	param.testEnd = 0;					/*16*/
	param.testNumber = 0;				/*17*/
	param.costFunction = "";			/*18*/

	if ( fnn.fail() ){
		cout<< "fail to open nn file : " << path<< endl;
		return false;
	}

	fnn >> ws;
	while( getline(fnn,in) ){
		removeChar(in, (char)13);
		replaceChar(in, '\t', ' ');
		removeChar(in, ' ');
		if(in[0] == '#') continue;
		if(in[0] == '\0') continue;
		//cout << " readnn : " << in;
		auto sline = split(in,'=');
		if( sline.size() != 2 ) continue;
		in = sline[0];
		out = sline[1];
		if(in == "sampleType"){
			if( out == "classification" )
				param.sampleType = nn_t::classification;
			else if( out == "regression" )
				param.sampleType = nn_t::regression;
			else if( out == "timeseries" )
				param.sampleType = nn_t::timeseries;
			else{
				errorString(" tpye wrong ", out ,"");
				return false;
			}
		}else if(in == "stopTrainingCost"){
			if( !isFloat(out)) return false;
			param.stopTrainingCost = atof(out.c_str());
		}else if(in == "trainFeature"){
			if( !isInt(out)) return false;
			param.trainFeature = atof(out.c_str());
		}else if(in == "sampleData"){
			param.sampleData = out.substr(1,out.size()-2);
		}else if(in == "iteration"){
			if( !isInt(out) ) return false;
			param.iteration = atoi(out.c_str());
		}else if(in == "learningRate"){
			if( !isFloat(out) ) return false;
			param.learningRate = atof(out.c_str());
		}else if(in == "hidden"){
			if(!readLayerParam_hidden(out, lparam)){
				errorString(" readnn error ", out,"");
				return false;
			}
			param.hidden.push_back(lparam);
		}else if(in == "output"){
			if(!readLayerParam_output(out, param.output)){
				errorString(" readnn error ", out,"");
				return false;
			}
		}else if(in == "normalizeMethod"){
			param.normalizeMethod = out;
		}else if(in == "loadWeight"){
			if(out == "true") param.loadWeight = true;
			param.loadWeight = false;
		}else if(in == "saveWeight"){
			if(out == "true") param.saveWeight = true;
			param.saveWeight = false;
		}else if(in == "weightPath"){
			param.weightPath = out;
		}else if(in == "weightName"){
			param.weightName = out;
		}else if(in == "defaultActivation"){
			if(out.size() == 0)
				cout << " readnn error : default activation empty" << endl;
			param.defaultActivation = out;
		}else if(in == "featureOffset"){
		}else if(in == "trainType"){
			if(!readnnSampleSet(out, param.trainType, param.trainStart, param.trainEnd, param.trainNumber)){
				errorString(" readnn trainSample argument error",out,"");
				return false;
			}
		}else if(in == "testType"){
			if(!readnnSampleSet(out, param.testType, param.testStart, param.testEnd, param.testNumber)){
				errorString(" readnn testSample argument error",out,"");
				return false;
			}
		}else if(in == "testStep"){
			if(!isInt(out)) return false;
			param.testStep = atoi(out.c_str());
		}else if(in == "costFunction"){
			param.costFunction = out;
		}else{
			errorString("no such parameter", out,"");
		}

	}
	fnn.close();
	return true;
}

