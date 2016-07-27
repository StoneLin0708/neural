#include "load/include/StringProcess.hpp"
#include "load/include/Loader.hpp"
#include "method/include/Normailze.hpp"
#include "load/include/SampleFeeder.hpp"
#include <armadillo>
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

        string normalizeMethod;
        bool loadWeight;
        bool saveWeight;
        string weightPath;

        string weightName;
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
    }Param;
*/

    bool nnFileRead(const string& path, nnFile_t& nnf){
        ifstream fnn;
        fnn.open(path.c_str(), ios::in);
        if(!fnn) return false;
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
        //input layer
        if( !isInt( mp["InputLayer"] )) return false;
        n.Layer.push_back( new InputLayer( atof(mp["InputLayer"].c_str()) ) );

        if(!isDouble( mp["LearningRate"] )) {delete static_cast<InputLayer*>(n.Layer[0]);n.Layer.clear();return false;}
        double LR = atof(mp["LearningRate"].c_str());
        //hidden layer
        int hidden=1;
        stringstream ss;
        string s;
        bool success = true;
        while(true){
            ss.clear();
            ss << "HiddenLayer" << hidden;
            ss >> s;
            auto sp = split( mp[s], ',');
            if( sp.size() == 0) break; if( sp.size() != 2) success = false;
            if(!isInt(sp[0])) success = false;
            auto act = fun::find_act( sp[1] ); if(!get<2>(act)) success =  false;

            if(success){
                n.Layer.push_back( new HiddenLayer(
                                hidden, atoi(sp[0].c_str()), n.Layer.back()->Nodes, LR,
                                get<0>(act), get<1>(act) ) );
            }else{
                for(int i=n.Layer.size()-1;i>0;--i){
                    delete static_cast<HiddenLayer*>(n.Layer[i]);
                }
                delete static_cast<InputLayer*>(n.Layer[0]);
                n.Layer.clear();
                return false;
            }
            ++hidden;
        }
        //output layer
        auto sp = split( mp["OutputLayer"], ',');
        if( sp.size() != 2) success = false;
        if(!isInt(sp[0])) success = false;
        auto act = fun::find_act( sp[1] ); if(!get<2>(act)) success = false;
        auto cost = fun::find_cost( mp["CostFunction"] ); if(!get<2>(cost)) success = false;
        if(!success){
                for(int i=n.Layer.size()-1;i>0;--i){
                    delete static_cast<HiddenLayer*>(n.Layer[i]);
                }
                delete static_cast<InputLayer*>(n.Layer[0]);
                n.Layer.clear();
                return false;
        }
        n.Layer.push_back( new OutputLayer(
                            hidden, atoi(sp[0].c_str()), n.Layer.back()->Nodes, LR,
                            get<0>(act), get<1>(act),
                            get<0>(cost), get<1>(cost) ) );
        for(int i=n.Layer.size()-1;i>0;--i)
            static_cast<CalLayer*>(n.Layer[i])->RandomWeight(-4,4);
        return true;
    }

    bool loadSample(nnFile_t &mp, Sample &s, string type){
        if(!s.read(mp[type])) return false;
        if(mp["SampleType"] == "Classification"){
            //cout << "s i:"<<endl<<s.input << endl;
            //cout << "s o:"<<endl<<s.output << endl;
            s.norm_in = Normalize(s.input,-1,1);
            auto out = ReMapping(s.output);
            if(!get<1>(out)) return false;
            s.outputMap = get<0>(out);
            s.n_output = s.output.n_cols;
            //cout << "s i n:"<<endl<<s.input << endl;
            //cout << "s o n:"<<endl<<s.output << endl;
        }else{
            Normalize(s.input,-1,1);
            Normalize(s.output,0,1);
        }
        return true;
    }

    bool loadTrain(nnFile_t &mp, Trainer &t){
        if( !isDouble( mp["Iteration"] ) ) return false;
        t.iteration = atof( mp["Iteration"].c_str() );
        return true;
    }

}
