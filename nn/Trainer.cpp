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
            //cout << static_cast<OutputLayer*>(n->Layer.back())->cost;
            n->update();
            sf->reset();
        }
    }

    void Trainer::set(Network *n, Sample *s){
        this->n = n;
        sf = new SampleFeeder(s, &n->Layer[0]->out, &static_cast<OutputLayer*>(n->Layer.back())->desire);
        //cout << *static_cast<CalLayer*>(n->Layer[1]);
        //cout << *static_cast<CalLayer*>(n->Layer[2]);
    }

    bool Trainer::gradientChecking(){
        double delta = 1.0e-4;
        double errorMax = 1.0e-6;
        double fw,fwd;
        auto lout = static_cast<OutputLayer*>(n->Layer.back());

        for(int l=n->Layer.size()-1; l>0; --l){
            cout << "Gradient Checking Layer : " << l << endl;
            auto layer = static_cast<CalLayer*>(n->Layer[l]);
            //compute delta weight
            sf->reset();
            while(!sf->isLast()){
                sf->next();
                n->fp();
                n->bp();
            }

            for(int i=0; i<layer->Inputs; ++i){
                for(int j=0; j<layer->Nodes; ++j){
                    fw = -1 * layer->wupdates(i,j)/layer->wupdateCounter;

                    layer->weight(i,j) += delta;
                    fwd = 0;
                    sf->reset();
                    while(!sf->isLast()){
                        sf->next();
                        n->fp();
                        auto c = lout->fcost(lout->desire,lout->out,lout->Nodes);
                        for(int k=0;k<lout->Nodes;++k)
                            fwd += c(k);
                    }

                    layer->weight(i,j) -= delta*2;
                    sf->reset();
                    while(!sf->isLast()){
                        sf->next();
                        n->fp();
                        auto c = lout->fcost(lout->desire,lout->out,lout->Nodes);
                        for(int k=0;k<lout->Nodes;++k)
                            fwd -= c(k);
                    }
                    layer->weight(i,j) += delta;
                    fwd /= 2 * delta * layer->wupdateCounter;

                    //cout << j << ',' << i << ",fw="<<fw<<",fwd="<< fwd<<endl;
                    if( fabs(fwd - fw) > errorMax ){
                        cout<< "gradient check fail"<< " Layer "<< l<<"("<<i<<","<<j<<")"<<endl;
                        cout<< "fwd " << fwd<< " fw " <<fw<<endl;
                        //return false;
                    }
                }
            }
            for(int l=n->Layer.size()-1; l>0; --l){
                auto layer = static_cast<CalLayer*>(n->Layer[l]);
                layer->wupdateCounter = 0;
                layer->wupdates.zeros();
            }
        }
        cout << "gradient check success "<<endl;
        return true;
    }
}

