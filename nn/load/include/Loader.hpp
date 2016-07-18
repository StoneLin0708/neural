#include "core/include/nn.hpp"
#include "method/include/method.hpp"
#include "load/include/sample.hpp"

#include <string>

using std::string;

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

namespace  nn{

    bool loadNetwork(vector<string> file, Network &n);
    bool loadSample(vector<string> file, Sample &train, Sample &test);
    void loadTrain();

}
