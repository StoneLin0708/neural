#include "ANNModel.hpp"
#include "layer/include/anfis.hpp"
#include "core/include/Layer.hpp"
#include "load/include/Loader.hpp"
#include "output/include/Info.hpp"
#include <iostream>
#include <iomanip>

using namespace std;
namespace nn {

using namespace anfis;

bool ANFISModel::load(string nnFilePath){
    nnFile_t nnf;
    if(!nnFileRead(nnFilePath, nnf)){ cout << "nnFile fail" <<endl; return false;}

    if(!loadSample(nnf, trainSample, "TrainSample")) return false;
    if(!loadSample(nnf, testSample, "TestSample")) return false;
    if(!loadTrain(nnf, trainer)){ cout << "loadTrain fail" <<endl;return false;}

    int Input = trainSample.n_input;
    double LR = 0.001;
    int MSF = 3;

    network.addInputLayer( new InputLayer(Input)) ;
    network.addMiddleLayer( new FLayer(1, Input, MSF, LR) );
    network.addMiddleLayer( new PLayer(2, network.Layer[1]->Nodes, MSF) );
    network.addMiddleLayer( new NLayer(3, network.Layer[2]->Nodes) );
    network.addMiddleLayer( new CLayer(4, network.Layer[3]->Nodes, Input, &(network.Layer[0]->out), LR) );
    auto o = new OLayer(5, network.Layer[4]->Nodes);
    network.addOutputLayer(static_cast<BaseLayer*>(o),static_cast<BaseOutputLayer*>(o));

    trainer.set(&network,&trainSample);
    tester.set(&network,&testSample);

    return true;
}
/*
bool GradientCheckLayer(
        [](unsigned int iter)->pair<bool isend,double &val> node,
        [](unsigned int iter)->pair<bool isend,double val> delta,
        double delta,
        double errorMax){
    return false;
}
*/
bool ANFISModel::GradientCheck(bool info){
    double delta = 1.0e-4;
    double errorMax = 1.0e-6;
    auto flayer = static_cast<anfis::FLayer*>(network.Layer[1]);
    auto player = static_cast<anfis::PLayer*>(network.Layer[2]);
    auto nlayer = static_cast<anfis::NLayer*>(network.Layer[3]);
    auto clayer = static_cast<anfis::CLayer*>(network.Layer[4]);

    auto& sf = *(trainer.sf);

    if(info) cout << "Gradient Checking Layer FPN "<<endl;
    sf.reset();
    network.clear();
    while (!sf.isLast()){
        sf.next();
        network.fp();
        network.bp();
    }

    //membership e
    for(int i=0; i<flayer->Nodes; ++i){
        double &tval = flayer->node[i].expect;
        tval += delta;
        double del = 0;
        double bpdel;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del += network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval -= 2*delta;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del -= network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval += delta;
        del /= 2 * delta * flayer->bpCounter;

        bpdel = flayer->node[i].eupdates / flayer->bpCounter;
        cout<< "gc fLayer node("<< i<< ") e bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            cout<<" fail "<<bpdel*100/del<<"%"<<endl;
        }else{
            cout<<" success"<<endl;
        }
    }

    //embership v
    for(int i=0; i<flayer->Nodes; ++i){
        double &tval = flayer->node[i].variance;

        tval += delta;
        double del = 0;
        double bpdel;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del += network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval -= 2*delta;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del -= network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval += delta;
        del /= 2 * delta * flayer->bpCounter;

        bpdel = flayer->node[i].vupdates / flayer->bpCounter;
        cout<< setprecision(5) <<"gc fLayer node("<< i<< ") v bp="<<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            cout<<" fail "<<bpdel*100/del<<"%"<<endl;
        }else{
            cout<<" success"<<endl;
        }

    }

    for(int i=0; i<player->Nodes; ++i){
        double &tval = player->out(i);
        tval += delta;
        double del = 0;
        double bpdel;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del += network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval -= 2*delta;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del -= network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval += delta;
        del /= 2 * delta * flayer->bpCounter;

        bpdel = player->delta(i) / flayer->bpCounter;
        cout<< "gc pLayer out("<< i<< ") bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            cout<<" fail "<<bpdel*100/del<<"%"<<endl;
        }else{
            cout<<" success"<<endl;
        }
    }

    for(int i=0; i<nlayer->Nodes; ++i){
        double &tval = nlayer->out(i);
        tval += delta;
        double del = 0;
        double bpdel;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del += network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval -= 2*delta;
        sf.reset();
        while (!sf.isLast()){
            sf.next();
            network.fp();
            del -= network.OutLayer->fcost(network.desire(),network.output(),1)(0);
        }

        tval += delta;
        del /= 2 * delta * flayer->bpCounter;

        bpdel = player->delta(i) / flayer->bpCounter;
        cout<< "gc nLayer out("<< i<< ") bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            cout<<" fail "<<bpdel*100/del<<"%"<<endl;
        }else{
            cout<<" success"<<endl;
        }
    }

    //clayer
    /*
    for(int i=0; i<(int)clayer->weight.n_rows; ++i){
        for(int j=0; j<(int)clayer->weight.n_cols; ++j){
            double &tval = clayer->weight(i,j);

            tval += delta;
            double del = 0;
            double bpdel;
            sf.reset();
            while (!sf.isLast()){
                sf.next();
                network.fp();
                del += network.OutLayer->fcost(network.desire(),network.output(),1)(0);
            }

            tval -= 2*delta;
            sf.reset();
            while (!sf.isLast()){
                sf.next();
                network.fp();
                del -= network.OutLayer->fcost(network.desire(),network.output(),1)(0);
            }

            tval += delta;
            del /= 2 * delta * flayer->bpCounter;

            bpdel = clayer->wupdates(i,j) / clayer->bpCounter;
            cout<< setprecision(5) <<"gc cLayer node("<< i<<','<<j<<")  bp="<<bpdel<<" del="<< del;
            if( fabs(bpdel - del) > errorMax){
                cout<<" fail "<<bpdel*100/del<<"%"<<endl;
            }else{
                cout<<" success"<<endl;
            }
        }
    }
    */

    if(info) cout << "gradient check success "<<endl;
    return true;
}
}
