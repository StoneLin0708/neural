#include "load/include/stringProcess.hpp"
#include "load/include/Loader.hpp"

#include "load/include/sampleFeeder.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <float.h>
#include <math.h>

using namespace std;

namespace nn{
/* Param
    typedef struct Param{
        nn_t::output_t sampleType;
        double stopTrainingCost;
        double trainFeature;
        string sampleData;

        int iteration;
        double learningRate;
        vector<struct layerParam> hidden;
        struct layerParam output;

        string normalizeMethod;
        bool loadWeight;
        bool saveWeight;
        string weightPath;

        string weightName;
        string defaultActivation;
        int featureOffset;

        sampleSet::type trainType;
        int trainStart;
        int trainEnd;
        int trainNumber;

        sampleSet::type testType;
        int testStart;
        int testEnd;
        int testNumber;

        int testStep;
        string costFunction;
    }Param;
*/
    bool nnLoad(const string& path, Network &n, Sample &){
        nnFile_t nnf;
        nnFileRead(path, nnf);
        loadNetwork(nnf, n);
        //loadSample();
        return true;
    }

    bool nnFileRead(const string& path, nnFile_t& nnf){
        ifstream fnn;
        fnn.open(path.c_str(), ios::in);
        string in;
        while( getline(fnn,in) ){
            removeChar(in,'\n');
            replaceChar(in, '\t', ' ');
            removeChar(in, ' ');
            if(in[0] == '#' ||in[0] == '\0') continue;
            auto sp = split(in,'='); if(sp.size() != 2){fnn.close();return false;}
            auto re  = nnf.insert( make_pair(sp[0], sp[1]) );
            if( !re.second ){fnn.close(); return false;}
        }
        fnn.close();
        return true;
    }

    bool loadNetwork(nnFile_t &mp, Network &n){
        if( !isInt( mp["InputLayer"] )) return false;
        n.Layer.push_back( InputLayer( atof(mp["InputLayer"].c_str()) ) );

        if(!isDouble( mp["LearningRate"] )) return false;
        double LR = atof(mp["LearningRate"].c_str());
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
            auto act = fun::find_act( sp[1] ); if(!get<2>(act)) return false;
            n.Layer.push_back(HiddenLayer(
                                hidden, atoi(sp[0].c_str()), n.Layer.back().Nodes, LR,
                                get<0>(act), get<1>(act) ) );
        }
        auto sp = split( mp["OutputLayer"], ',');
        if( sp.size() == 0) return false; if( sp.size() != 2) return false;
        if(!isInt(sp[0])) return false;
        auto act = fun::find_act( sp[1] ); if(!get<2>(act)) return false;
        auto cost = fun::find_cost( mp["CostFunction"] ); if(!get<2>(cost)) return false;
        n.Layer.push_back(OutputLayer(
                            hidden, atoi(sp[0].c_str()), n.Layer.back().Nodes, LR,
                            get<0>(act), get<1>(act),
                            get<0>(cost), get<1>(cost) ) );
        return true;
    }

    bool loadSample(nnFile_t &mp, Sample &train, Sample &test){
        if( !train.read(mp["TrainSample"]) ) return false;
        if( !test.read(mp["TestSample"]) ) cout << "no test data" << endl;
        return true;
    }


}
