#include "ANNModel.hpp"
#include "core/include/AnfisLayer.hpp"
#include "core/include/Layer.hpp"
#include "load/include/Loader.hpp"
#include "output/include/Info.hpp"
#include <iostream>
using namespace std;
namespace nn {

using namespace anfis;

bool ANFISModel::load(string &nnFilePath){
    nnFile_t nnf;
    if(!nnFileRead(nnFilePath, nnf)){ cout << "nnFile fail" <<endl; return false;}

    if(!loadSample(nnf, trainSample, "TrainSample")) return false;
    if(!loadSample(nnf, testSample, "TestSample")) return false;
    if(!loadTrain(nnf, trainer)){ cout << "loadTrain fail" <<endl;return false;}

    auto &Layer = network.Layer;
    Layer.push_back(new InputLayer(trainSample.n_input));
    Layer[0]->out = arma::zeros<rowvec>(trainSample.n_input);
    Layer.push_back(new FuzzyLayer(1, 2, Layer[0]->Nodes-1, 2, 1));
    Layer[1]->out = arma::zeros<rowvec>(2);
    Layer.push_back(new OutputLayer(2, 1, Layer[1]->Nodes-1, 1,
                    fun::sigmoid,fun::dsigmoid,fun::mse,fun::dmse) );

    trainer.set( &network, &trainSample);
    tester.set( &network, &testSample);
    showNetwork( network );

    return true;
}

}
