#include "nnio.hpp"
#include "stringCheck.hpp"
#include <fstream>
#include <vector>
#include <float.h>

using namespace std;

bool readnnTestSample(const string &in,nnParam &param){
	vector<string> sp = split(in,',');
	if( sp[0] == "all" ){
		param.testSampleType = nn_t::all ;
		param.testSampleStart = 1 ;
		param.testSampleNumber = -1;
		return true;
	}
	else if( sp[0] == "number" ){
		if(sp.size() != 3) return false;
		param.testSampleType = nn_t::number ;
		if( !isInt(sp[1]) || !isInt(sp[2]) ) return false;
		param.testSampleStart = atoi(sp[1].c_str());
		param.testSampleEnd = atoi(sp[2].c_str());
		param.testSampleNumber = param.testSampleEnd -
			param.testSampleStart + 1;
		return true;
	}

	errorString(" readnn fail sampling error ", in,"");
	return false;
}

bool readnnSampling(const string &in,nnParam &param){
	vector<string> sp = split(in,',');
	if( sp[0] == "all" ){
		param.samplingType = nn_t::all ;
		param.samplingStart = 1 ;
		param.samplingNumber = -1;
		return true;
	}
	else if( sp[0] == "number" ){
		if(sp.size() != 3) return false;
		param.samplingType = nn_t::number ;
		if( !isInt(sp[1]) || !isInt(sp[2]) ) return false;
		param.samplingStart = atoi(sp[1].c_str());
		param.samplingEnd = atoi(sp[2].c_str());
		param.samplingNumber = param.samplingEnd -
			param.samplingStart + 1;
		return true;
	}
	else if( sp[0] == "bunch" ){
		param.samplingType = nn_t::bunch ;
		param.samplingStart = 1 ;
		if(sp.size() != 3) return false;
		if(!isInt(sp[1])) return false;
		param.samplingNumber = atoi( sp[1].c_str() );
		if(!isInt(sp[2],false) ){
			if( sp[2] != "all" ) return false;
			param.samplingEnd = -1;
		}
		else
			param.samplingEnd = atoi( sp[2].c_str() ) * param.samplingNumber;
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

int readnnWhich(const string &in, string &out){
	string strList[] = {
		"sampleType",		/*0*/
		"stopTrainingCost",		/*1*/
		"notUsing",		/*2*/
		"sampleData",		/*3*/

		"iteration",		/*4*/
		"learningRate",		/*5*/
		"hidden",			/*6*/
		"output",			/*7*/

		"normalizeMethod",	/*8*/
		"loadWeight",		/*9*/
		"saveWeight",		/*10*/
		"weightPath",		/*11*/

		"weightName",		/*12*/
		"defaultActivation",/*13*/
		"testOnly",			/*14*/
		"samplingType",		/*15*/

		"testType",			/*16*/
		"testStep"			/*17*/
	};

	vector<string> sstr = split(in,'=');
	if(sstr.size() != 2){
		out = in;
		return -1;
	}
	for(int i=0; i<18; ++i)
		if( sstr[0] == strList[i] ){
			out = sstr[1];
			return i;
		}
	out = sstr[0];
	return -1;
}

bool readnn(const string& path, nnParam &param){
	ifstream fnn;
	fnn.open(path.c_str(), ios::in);
	string in,out;
	layerParam lparam;

	param.sampleType = nn_t::empty;			/*0*/
	param.stopTrainingCost = -DBL_MAX;		/*1*/
	param.outputScale = 1;					/*2*/
	param.sampleData = "";					/*n 3*/

	param.iteration = 0;					/*n 4*/
	param.learningRate = 0;					/*n 5*/
	//param.hidden;							/*6*/
	//param.output;							/*n 7*/

	param.normalizeMethod = "";				/*8*/
	param.loadWeight = false;				/*9*/
	param.saveWeight = false;				/*10*/
	param.weightPath = "";					/*11*/

	param.weightName = "";					/*12*/
	param.defaultActivation = "";			/*n 13*/
	param.testOnly =false;					/*14*/

	param.samplingType = nn_t::all;	/*15*/

	param.samplingStart = 0;				/*15*/
	param.samplingEnd = 0;					/*15*/
	param.samplingNumber = 0;				/*15*/

	param.testSampleType = nn_t::all;		/*16*/
	param.testSampleStart = 0;				/*16*/
	param.testSampleEnd = 0;				/*16*/
	param.testSampleNumber = 0;				/*17*/

	if ( fnn.fail() ){
		cout<< "fail to open nn file : " << path<< endl;
		return false;
	}

	while( fnn >> in ){
		if(in[0] == '#') continue;
		cout << " readnn : " << in << " ... ";
		switch(readnnWhich(in, out) ){
		case -1:
			errorString("no such parameter", out,"");
			break;
		case 0:
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
			break;
		case 1:
			if( !isFloat(out)) return false;
			param.stopTrainingCost = atof(out.c_str());
			break;
		case 2:
			break;
		case 3:
			param.sampleData = out.substr(1,out.size()-2);
			break;
		case 4:
			if( !isInt(out) ) return false;
			param.iteration = atoi(out.c_str());
			break;
		case 5:
			if( !isFloat(out) ) return false;
			param.learningRate = atof(out.c_str());
			break;
		case 6:
			if(!readLayerParam_hidden(out, lparam)){
				errorString(" readnn error ", out,"");
				return false;
			}
			param.hidden.push_back(lparam);
			break;
		case 7:
			if(!readLayerParam_output(out, param.output)){
				errorString(" readnn error ", out,"");
				return false;
			}
			break;
		case 8:
			param.normalizeMethod = out;
			break;
		case 9:
			if(out == "true") param.loadWeight = true;
			param.loadWeight = false;
			break;
		case 10:
			if(out == "true") param.saveWeight = true;
			param.saveWeight = false;
			break;
		case 11:
			param.weightPath = out;
			break;
		case 12:
			param.weightName = out;
			break;
		case 13:
			if(out.size() == 0)
				cout << " readnn error : default activation empty" << endl;
			param.defaultActivation = out;
			break;
		case 14:
			break;
		case 15:
			if(!readnnSampling(out, param)){
				errorString(" readnn sampling argument error",out,"");
				return false;
			}
			break;
		case 16:
			if(!readnnTestSample(out, param)){
				errorString(" readnn test argument error",out,"");
				return false;
			}
			break;
		case 17:
			if(!isInt(out)) return false;
			param.testStep = atoi(out.c_str());
			break;
		}
		cout << endl;
	}
	fnn.close();
//------------------logic check-------------------------
	cout<< " nn file logic check ..." << endl;
	if( param.testOnly && (
		(param.sampleType == nn_t::empty) ||
		param.sampleData.empty() ||
		(param.output.nodes == 0 ) ||
		param.defaultActivation.empty()
		) ){
		cout << " readnn error : necessary param missing , check readme" << endl;
		return false;
	}

	if( param.sampleType == nn_t::empty){
		cout << " readnn error : sampleType necessary " << endl;
		return false;
	}
	else if( param.sampleData.empty() ){
		cout << " readnn error : sampleData necessary " << endl;
		return false;
	}
	else if( param.iteration == 0){
		cout << " readnn error : iteration necessary " << endl;
		return false;
	}
	else if( param.learningRate == 0){
		cout << " readnn error : learningRate necessary " << endl;
		return false;
	}
	else if( param.output.nodes == 0){
		cout << " readnn error : output necessary " << endl;
		return false;
	}
	else if( param.defaultActivation.empty()){
		cout << " readnn error : defaultActivatoin necessary " << endl;
		return false;
	}
	cout << " nn file logic check ... done" << endl;
	return true;
}
