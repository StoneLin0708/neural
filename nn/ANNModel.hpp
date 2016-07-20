#pragma once
#include "core/include/nn.hpp"
#include "load/include/Sample.hpp"
#include "trainer.hpp"
#include <string>

namespace  nn{

class ANNModel
{
public:
    ANNModel();

    bool load(std::string &nnFilePath);

    Network network;
    Sample trainSample;
    Trainer trainer;

};

}
