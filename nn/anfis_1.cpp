#include "ANNModel.hpp"
#include "layer/include/anfis.hpp"
#include "core/include/Layer.hpp"
#include "load/include/Loader.hpp"
#include "output/include/Info.hpp"
#include <iostream>
using namespace std;
namespace nn {

using namespace anfis;

bool ANFISModel::load(string nnFilePath){
    nnFile_t nnf;
    if(!nnFileRead(nnFilePath, nnf)){ cout << "nnFile fail" <<endl; return false;}

    if(!loadSample(nnf, trainSample, "TrainSample")) return false;
    //if(!loadSample(nnf, testSample, "TestSample")) return false;
    if(!loadTrain(nnf, trainer)){ cout << "loadTrain fail" <<endl;return false;}

    int Input = trainSample.n_input;
    double LR = 0.001;
    int MSF = 3;

    network.addInputLayer( new InputLayer(Input)) ;
    network.addMiddleLayer( new FPNLayer(1, Input, MSF, LR) );
    network.addMiddleLayer( new CLayer(2, network.Layer[1]->Nodes, Input, &(network.Layer[0]->out), LR) );
    auto o = new OLayer(3, Input);
    network.addOutputLayer(static_cast<BaseLayer*>(o),static_cast<BaseOutputLayer*>(o));

    trainer.set(&network,&trainSample);

    return true;
}

}
