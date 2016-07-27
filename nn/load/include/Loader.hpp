#include "core/include/nn.hpp"
#include "method/include/Method.hpp"
#include "load/include/Sample.hpp"
#include "Trainer.hpp"

#include <string>
#include <map>

/*
namespace nn_t{

    typedef enum{
        empty = 0,
        classification,
        regression,
        timeseries
    }output_t;

}
*/

namespace nn{
    typedef std::map<std::string,std::string> nnFile_t;

    bool nnFileRead(const string& path, nnFile_t&);

    bool loadNetwork(nnFile_t&, Network &);
    bool loadSample(nnFile_t&, Sample&, string type);
    bool loadTrain(nnFile_t&, Trainer&);

}
