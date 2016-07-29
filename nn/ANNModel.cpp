#include "ANNModel.hpp"
#include "load/include/Loader.hpp"
#include "output/include/Info.hpp"

#include <iostream>
using namespace std;
namespace  nn{

ANNModel::ANNModel()
{

}

bool ANNModel::load(string &nnFilePath)
{
    nnFile_t nnf;
    if(!nnFileRead(nnFilePath, nnf)){ cout << "nnFile fail" <<endl; return false;}

    if(!loadNetwork(nnf, network)){ cout << "loadNetwork fail" <<endl; return false;}
    if(!loadSample(nnf, trainSample, "TrainSample")) return false;
    if(!loadSample(nnf, testSample, "TestSample")) return false;
    if(!loadTrain(nnf, trainer)){ cout << "loadTrain fail" <<endl;return false;}

    trainer.set( &network, &trainSample);
    tester.set( &network, &testSample);
    showNetwork( network );

    return true;
}

}
