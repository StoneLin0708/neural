#include "Trainer.hpp"
#include "core/include/Layer.hpp"
#include <iostream>
#include <iomanip>
#include "Timer.hpp"

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
    Timer timer, timerPredict;
    bool calCost = false;
    timer.start();
    timerPredict.start();
    sf->reset();
    for(int i=iteration-1; i>=0; --i){
        n->clear();
        while(!sf->isLast()){
            sf->next();
            n->fp();
            if(calCost) NN_GET_OUTPUT_LAYER(*n)->CalCost();
            n->bp();
            /*
            cout << "o" <<endl << n->Layer[0]->out<<endl;
            cout << *static_cast<CalLayer*>(n->Layer[1]);
            cout << *static_cast<CalLayer*>(n->Layer[2]);
            cout << static_cast<OutputLayer*>(n->Layer.back())->desire;
            cout << "-------" <<endl;
            cin.get();
            */
        }
        /*
        cout << *static_cast<CalLayer*>(n->Layer[1]);
        cout << *static_cast<CalLayer*>(n->Layer[2]);
        //cout << *static_cast<CalLayer*>(n->Layer[3]);
        cin.get();
        */
        if(calCost){
            cout<< "\rcost :" << setw(12) << fixed<< setprecision(10)
                << mean(NN_GET_OUTPUT_LAYER(*n)->costs/NN_GET_OUTPUT_LAYER(*n)->fpCounter)
                << " time left : " << setw(8)<<fixed<< setprecision(1)
                << (timerPredict.countMS() / 1000) * i / (iteration - i) <<" sec ";
            cout.flush();

            timer.start();
            calCost = false;
        }

        if(timer.countMS() > 500) calCost = true;

        n->update();
        sf->reset();
    }
    cout << endl;
}

void Trainer::set(Network *n, Sample *s){
    this->n = n;
    sf = new SampleFeeder(s, &n->Layer[0]->out, &(NN_GET_OUTPUT_LAYER(*n)->desire));
}

}
