#include "Tester.hpp"
#include "layer/include/feedforward.hpp"
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
    sf = new SampleFeeder(s, &n->input(), &n->desire());
}

void Tester::testClassification(){
    int right=0;
    sf->reset();
    n->clear();
    while(!sf->isLast()){
        sf->next();
        n->fp();
        n->OutLayer->CalCost();
        if(n->output().index_max()== n->desire().index_max())
            ++right;
    }
    cout<< "Test Cost : "
        << mean(n->OutLayer->costs/static_cast<CalLayer*>(n->Layer.back())->fpCounter)<<endl
        << "Classification Accurate : " << fixed << setprecision(2)
        << 100 * (double)right / static_cast<CalLayer*>(n->Layer.back())->fpCounter <<endl;
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
            n->OutLayer->CalCost();
            /*
        cout<<"----"<<endl;
        cout<<n->input();
        cout<<n->output();
        cout<<n->desire();
        cout<<"----"<<endl;
        cin.get();
        */
        }
        cout<< "cost :"
            << mean(n->OutLayer->costs/static_cast<CalLayer*>(n->Layer.back())->fpCounter)<<endl;
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

    for(int l=n->Layer.size()-1; l>0; --l){
        if(info) cout << "Gradient Checking Layer : " << l << endl;
        auto layer = static_cast<feedforward::FeedForwardCalLayer*>(n->Layer[l]);
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
                    auto c = n->OutLayer->fcost(n->desire(),n->output(),n->Layer.back()->Nodes);
                    for(int k=0;k<n->Layer.back()->Nodes;++k)
                        fwd += c(k);
                }

                layer->weight(i,j) -= delta*2;
                sf->reset();
                while(!sf->isLast()){
                    sf->next();
                    n->fp();
                    auto c = n->OutLayer->fcost(n->desire(),n->output(),n->Layer.back()->Nodes);
                    for(int k=0;k<n->Layer.back()->Nodes;++k)
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
    }
    if(info) cout << "gradient check success "<<endl;
    return true;
}


}
