#include "Tester.hpp"

#include "core/include/Layer.hpp"
#include <iostream>
#include <iomanip>

using namespace std;
namespace  nn {

Tester::Tester(){
    n = nullptr;
    sf = nullptr;
}

Tester::~Tester(){
       if(sf != nullptr) delete sf;
}

void Tester::set(Network *n, Sample *s){
    this->n = n;
    this->s = s;
    sf = new SampleFeeder(s, &n->Layer[0]->out, &(NN_GET_OUTPUT_LAYER(*n)->desire));
}

void Tester::testClassification(){
    int right=0;
    sf->reset();
    n->clear();
    while(!sf->isLast()){
        sf->next();
        n->fp();
        NN_GET_OUTPUT_LAYER(*n)->CalCost();
        if(NN_GET_OUTPUT_LAYER(*n)->out.index_max()==
                NN_GET_OUTPUT_LAYER(*n)->desire.index_max())
            ++right;
    }
    cout<< "Test Cost : "
        << mean(NN_GET_OUTPUT_LAYER(*n)->costs/NN_GET_OUTPUT_LAYER(*n)->fpCounter)<<endl
        << "Classification Accurate : " << fixed << setprecision(2)
        << 100 * (double)right / NN_GET_OUTPUT_LAYER(*n)->fpCounter <<endl;
}


void Tester::test(TestType type)
{
    switch (type) {
    case non:
        sf->reset();
        n->clear();
        while(!sf->isLast()){
            sf->next();
            n->fp();
            NN_GET_OUTPUT_LAYER(*n)->CalCost();
        }
        cout<< "cost :"
            << mean(NN_GET_OUTPUT_LAYER(*n)->costs/NN_GET_OUTPUT_LAYER(*n)->fpCounter)<<endl;
        break;
    case classification:
        testClassification();
        break;
    case regression:
        cout << "hi"<<endl;

        break;
    case timeseries:
        cout << "hi"<<endl;

        break;
    default:
        cout << "hi"<<endl;
        break;
    }
}


bool Tester::gradientChecking(bool info){
    double delta = 1.0e-4;
    double errorMax = 1.0e-6;
    double fw,fwd;
    auto lout = NN_GET_OUTPUT_LAYER(*n);

    for(int l=n->Layer.size()-1; l>0; --l){
        if(info) cout << "Gradient Checking Layer : " << l << endl;
        auto layer = static_cast<CalLayer*>(n->Layer[l]);
        sf->reset();
        n->clear();
        while(!sf->isLast()){
            sf->next();
            n->fp();
            n->bp();
        }

        for(int i=0; i<layer->Inputs; ++i){
            for(int j=0; j<layer->Nodes; ++j){
                fw = -1 * layer->wupdates(i,j)/layer->bpCounter;

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
                fwd /= 2 * delta * layer->bpCounter;

                if( fabs(fwd - fw) > errorMax ){
                    if(info){
                        cout<< "gradient check fail"<< " Layer "<< l<<"("<<i<<","<<j<<")"<<endl;
                        cout<< "fwd " << fwd<< " fw " <<fw<<endl;
                    }
                    return false;
                }
            }
        }
        /*
        for(int l=n->Layer.size()-1; l>0; --l){
            auto layer = static_cast<CalLayer*>(n->Layer[l]);
            layer->wupdateCounter = 0;
                layer->wupdates.zeros();
        }
        */
    }
    if(info) cout << "gradient check success "<<endl;
    return true;
}


}
