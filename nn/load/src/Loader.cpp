#include "load/include/stringProcess.hpp"
#include "load/include/Loader.hpp"

#include "load/include/sampleFeeder.hpp"

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <float.h>
#include <math.h>

using namespace std;

namespace nn{

    typedef struct Param{
        nn_t::output_t sampleType;			/*0*/
        double stopTrainingCost;			/*1*/
        double trainFeature;				/*2*/
        string sampleData;					/*3*/

        int iteration;						/*4*/
        double learningRate;				/*5*/
        vector<struct layerParam> hidden;	/*6*/
        struct layerParam output;			/*7*/

        string normalizeMethod;				/*8*/
        bool loadWeight;					/*9*/
        bool saveWeight;					/*10*/
        string weightPath;					/*11*/

        string weightName;					/*12*/
        string defaultActivation;			/*13*/
        int featureOffset;					/*14*/

        sampleSet::type trainType;			/*15*/
        int trainStart;						/*15*/
        int trainEnd;						/*15*/
        int trainNumber;					/*15*/

        sampleSet::type testType;				/*16*/
        int testStart;						/*16*/
        int testEnd;						/*16*/
        int testNumber;						/*16*/

        int testStep;						/*17*/
        string costFunction;				/*18*/
    }Param;

    bool nnRead(const string& path, Network &n, Sample &){
        ifstream fnn;
        fnn.open(path.c_str(), ios::in);
        string in;
        map<string,string> mp;
        while( getline(fnn,in) ){
            removeChar(in,'\n');
            replaceChar(in, '\t', ' ');
            removeChar(in, ' ');
            if(in[0] == '#' ||in[0] == '\0') continue;
            auto sp = split(in,'='); if(sp.size() != 2) return false;
            auto re  = mp.insert( sp[0], sp[0] );
            if( !re.second ) return false;
        }

        loadNetwork(mp, n);
        loadSample();

    }

    bool loadNetwork(map<string,string> &mp, Network &n){
        if( !isInt( mp["InputLayer"] )) return false;
        n.Layer.push_back( InputLayer( atof(mp["InputLayer"].c_str()) ) );

        int hidden=1;
        stringstream ss;
        string s;
        while(true){
            ss.clear();
            ss << "HiddenLayer" << hidden;
            ss >> s;
            auto sp = split( mp[s], ',');
            if( sp.size() == 0) break; if( sp.size() != 2) return false;
            if(!isInt(sp[0])) return false;
            auto act = funact::find( sp[1] ); if(!get<2>(act)) return false;
            n.Layer.push_back(HiddenLayer(
                                hidden, atoi(sp[0].c_str()), n.Layer.back().Nodes,
                                get<0>(act), get<1>(act) ) );
        }
        auto sp = split( mp["OutputLayer"], ',');
        if( sp.size() == 0) return false; if( sp.size() != 2) return false;
        if(!isInt(sp[0])) return false;
        auto act = funact::find( sp[1] ); if(!get<2>(act)) return false;
        n.Layer.push_back(OutputLayer(
                            hidden, atoi(sp[0].c_str()), n.Layer.back().Nodes,
                            get<0>(act), get<1>(act) ) );
    }

    bool loadSample(map<string,string> &mp, Sample &train, Sample &test){
        if( !train.read(mp["TrainSample"]) ) return false;
        if( !test.read(mp["TestSample"]) ) cout << "no test data" << endl;

    }


}
