#pragma once
#include "core/include/Layer.hpp"

#include <string>
#include <vector>

namespace nn{

    bool gradientChecking(int sample = -1);

    class Network{
    public:
        Network();
        ~Network();

        void fp(); //forward propagation
        void bp(); //backward propagation
        void update();

        void error(int &i);

        bool save(std::string path);

        std::vector<BaseLayer*> Layer;

        bool success(){return _init;}

    private:
        bool _init;

        int iteration;

    };

}
