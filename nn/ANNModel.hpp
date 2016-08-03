#pragma once
#include "core/include/network.hpp"
#include "load/include/Sample.hpp"
#include "Trainer.hpp"
#include "Tester.hpp"

#include <string>

namespace  nn{

class ANNModel
{
public:
    ANNModel();

    virtual bool load(std::string nnFilePath);

    Network network;
    Sample trainSample;
    Sample testSample;
    Trainer trainer;
    Tester tester;

};

class ANFISModel : public ANNModel{
public:
    bool load(std::string nnFilePath);

};

}
