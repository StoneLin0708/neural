#pragma once
#include "core/include/Layer.hpp"
#include <string>
#include <vector>

namespace nn{

    class Network{
    public:
        Network();
        virtual ~Network();

        void addInputLayer(BaseLayer *);
        void addMiddleLayer(BaseLayer *);
        void addOutputLayer(BaseLayer *, BaseOutputLayer *);

        virtual void clear();
        virtual void fp(); //forward propagation
        virtual void bp(); //backward propagation
        virtual void update();

        //virtual bool save(std::string path);

        rowvec& input(){return *ivec;}
        rowvec& output(){return *ovec;}
        rowvec& desire(){return *dvec;}

        std::vector<BaseLayer*> Layer;
        BaseOutputLayer *OutLayer;

    protected:
        rowvec *ivec;
        rowvec *ovec;
        rowvec *dvec;

    };

}
