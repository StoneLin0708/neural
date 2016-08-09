#include "Trainer.hpp"
#include "core/include/Layer.hpp"
#include <iostream>
#include <iomanip>
#include "output/Timer.hpp"
#include "layer/include/anfis.hpp"

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
    auto f = cout.flags();

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
            if(calCost) n->OutLayer->CalCost();
            n->bp();
        }
        if(calCost){
            cout<< "\rcost :" << setw(12) << fixed<< setprecision(10)
                << mean(n->OutLayer->costs/static_cast<CalLayer*>(n->Layer.back())->fpCounter)
                << " time left : " << setw(8)<<fixed<< setprecision(1)
                << (timerPredict.countMS() / 1000) * i / (iteration - i) <<" sec ";
            cout.flush();

            timer.start();
            calCost = false;
        }

        if(timer.countMS() > 200) calCost = true;

        n->update();
        sf->reset();
    }
    cout << endl;

    cout.flags(f);
}

void Trainer::set(Network *n, Sample *s){
    this->n = n;
    sf = new SampleFeeder(s, &(n->input()), &(n->desire())   );
}

}
