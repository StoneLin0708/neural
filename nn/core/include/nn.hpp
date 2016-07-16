#pragma once
//#define ARMA_NO_DEBUG
#include <vector>
#include <armadillo>
#include <string>

#include "sample.hpp"
#include "algorithm.hpp"
#include "sampleSet.hpp"

#include "core/include/Layer.hpp"

using std::string;
using std::vector;

namespace nn_t{

	struct activation{
		string name;
		double (*act)(double);
		double (*dact)(double);
	};

	typedef enum{
		empty = 0,
		classification,
		regression,
		timeseries
	}output_t;

}

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


    class Network{
    public:
        Network(nnParam param, bool info=false);

        //nnParam& getParam(){return _param;};
        bool enableParam();
        bool readSample(const string& path);

        bool gradientChecking(int sample = -1);
        void test(); //forward
        void train();
    /*
        void testResultClassification();
        void testResultRegression();
        void testResultSeries();
    */
        void error(int &i);

        vector<BaseLayer> Layer;

        rowvec features;
        rowvec outputs;
    /*
        int n_sample(){return _n_sample;};
        int n_feature(){return _n_feature;};

        int _n_sample;
        int _n_feature;
        int _n_output;
    */
        //nn_t::output_t type(){return _type;};
        nn_a::normParam featureNormParam;
        nn_a::normParam outputNormParam;

        vector<double> e;
        vector<double> en;
        bool success(){return _init;};

        double (*cost)(double desire, double out);
        double (*dcost)(double desire, double out);
    private:
        bool _init;

        void bp();
        void wupdate();
        void clear_wupdates();

        //nnParam _param;

        //nn_t::output_t _type;

        int iteration;

    };

}
