#pragma once
#include "core/include/Layer.hpp"

#include <string>
#include <vector>

//#define NN_GET_INPUT_LAYER( N ) static_cast<nn::InputLayer*>( (N).Layer[0] )
//#define NN_GET_HIDDEN_LAYER( N , L  ) static_cast<nn::HiddenLayer*>( (N).Layer[L-1] )
//#define NN_GET_HIDDEN_SIZE( N ) ( (int)(N).Layer.size()-2 )
//#define NN_GET_OUTPUT_LAYER( N ) static_cast<nn::OutputLayer*>( (N).Layer.back() )

//#define NN_GET_OUTPUT( N ) (NN_GET_OUTPUT_LAYER( N )->out)

namespace nn{

    class Network{
    public:
        Network();
        virtual ~Network();

        virtual void clear();
        virtual void fp(); //forward propagation
        virtual void bp(); //backward propagation
        virtual void update();

        //virtual bool save(std::string path);

        std::vector<BaseLayer*> Layer;

    };

}
