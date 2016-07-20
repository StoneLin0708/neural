#include "ANNModel.hpp"
#include "load/include/Loader.hpp"
namespace  nn{

ANNModel::ANNModel()
{

}

bool ANNModel::load(std::__cxx11::string &nnFilePath)
{
    return nnLoad(nnFilePath,network,trainSample,trainer);
}

}
