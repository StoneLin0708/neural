#include "trainer.hpp"
#include "core/include/Layer.hpp"
#include <iostream>
using namespace std;
namespace nn{

    Trainer::Trainer(){
        n = nullptr;
        sf = nullptr;
    }

    Trainer::~Trainer(){
       if(sf != nullptr) delete sf;
    }

    void Trainer::train()
    {
        sf->reset();
        for(int i=iteration-1; i>=0; --i){
            while(!sf->isLast()){
                sf->next();
                n->fp();
                n->bp();
                cout << *static_cast<CalLayer*>(n->Layer[1]);
                cout << *static_cast<CalLayer*>(n->Layer[2]);
                cin.get();
            }
            /*
            cout << *static_cast<CalLayer*>(n->Layer[1]);
            cout << *static_cast<CalLayer*>(n->Layer[2]);
            //cout << *static_cast<CalLayer*>(n->Layer[3]);
            cin.get();
            */
            n->update();
            sf->reset();
        }
    }

    void Trainer::set(Network *n, Sample *s){
        this->n = n;
        sf = new SampleFeeder(s, &n->Layer[0]->out, &static_cast<OutputLayer*>(n->Layer.back())->desire);
    }

}

