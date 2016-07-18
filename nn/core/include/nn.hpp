#pragma once
#include <vector>

#include "core/include/Layer.hpp"



namespace nn{

    bool gradientChecking(int sample = -1);

    class Network{
    public:
        Network();

        void test(); //forward
        void bp();
        void update();

        void error(int &i);

        std::vector<BaseLayer> Layer;

        bool success(){return _init;}

    private:
        bool _init;

        int iteration;

    };

}
