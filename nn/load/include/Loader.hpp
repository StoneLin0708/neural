#include "core/include/nn.hpp"
#include "method/include/method.hpp"
#include "load/include/sample.hpp"
#include "trainer.hpp"

#include <string>
#include <map>

namespace nn_t{

    struct activation{
        std::string name;
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
    typedef std::map<std::string,std::string> nnFile_t;

    bool nnLoad(const string& path, Network &, Sample &, Trainer &);

    bool nnFileRead(const string& path, nnFile_t&);

    bool loadNetwork(nnFile_t&, Network &n);
    bool loadSample(nnFile_t&, Sample &train, Sample &test);
    void loadTrain(nnFile_t&, Trainer&);

}
