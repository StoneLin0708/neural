#pragma once
#include "core/include/nn.hpp"
#include "load/include/Sample.hpp"
#include "Trainer.hpp"
#include "Tester.hpp"

#include <string>

namespace  nn{

class ANNModel
{
public:
    ANNModel();

    bool load(std::string &nnFilePath);

    Network network;
    Sample trainSample;
    Sample testSample;
    Trainer trainer;
    Tester tester;

};

}
