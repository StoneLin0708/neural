#include "Trainer.hpp"
#include "core/include/Layer.hpp"
#include <iostream>
#include <iomanip>
#include "Timer.hpp"
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
            cout << "-------------------------------------------------"<<endl;
            cout << n->Layer[0]->out<<endl;
            cout << "f "<<static_cast<anfis::FPNLayer*>(n->Layer[1])->fuzzy;
            cout << "p "<<static_cast<anfis::FPNLayer*>(n->Layer[1])->rule;
            cout << "n "<<static_cast<anfis::FPNLayer*>(n->Layer[1])->out;
            cout << "c "<<static_cast<anfis::CLayer*>(n->Layer[2])->out;
            cout << n->Layer[3]->out;
            cout << n->OutLayer->desire<<endl;
            cout << "dc "<<static_cast<anfis::CLayer*>(n->Layer[2])->delta;
            cout << " dn "<<static_cast<anfis::FPNLayer*>(n->Layer[1])->delta;
            //cin.get();
            if(calCost) n->OutLayer->CalCost();
            n->bp();
            /*
            cout << "o" <<endl << n->Layer[0]->out<<endl;
            cout << *static_cast<CalLayer*>(n->Layer[1]);
            cout << *static_cast<CalLayer*>(n->Layer[2]);
            cout << static_cast<OutputLayer*>(n->Layer.back())->desire;
            cout << "-------" <<endl;
            cin.get();
            */
            //cin.get();
        }
        /*
        cout << *static_cast<CalLayer*>(n->Layer[1]);
        cout << *static_cast<CalLayer*>(n->Layer[2]);
        //cout << *static_cast<CalLayer*>(n->Layer[3]);
        cin.get();
        */
        calCost = false;
        if(calCost){
            cout<< "\rcost :" << setw(12) << fixed<< setprecision(10)
                << mean(n->OutLayer->costs/static_cast<CalLayer*>(n->Layer.back())->fpCounter)
                << " time left : " << setw(8)<<fixed<< setprecision(1)
                << (timerPredict.countMS() / 1000) * i / (iteration - i) <<" sec ";
            cout.flush();

            timer.start();
            calCost = false;
        }

        if(timer.countMS() > 10) calCost = true;

        n->update();
        sf->reset();
    }
    cout << endl;
}

void Trainer::set(Network *n, Sample *s){
    this->n = n;
    sf = new SampleFeeder(s, &(n->input()), &(n->desire())   );
}

}
