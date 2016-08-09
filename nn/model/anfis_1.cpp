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

bool ANFISModel::GradientCheck(bool info){
    double delta = 1.0e-4;
    double errorMax = 1.0e-6;
    auto ilayer = static_cast<anfis::InputLayer*>(network.Layer[0]);
    auto flayer = static_cast<anfis::FLayer*>(network.Layer[1]);
    auto player = static_cast<anfis::PLayer*>(network.Layer[2]);
    auto nlayer = static_cast<anfis::NLayer*>(network.Layer[3]);
    auto clayer = static_cast<anfis::CLayer*>(network.Layer[4]);
    //auto olayer = static_cast<anfis::OLayer*>(network.Layer[5]);

    auto& sf = *(trainer.sf);

    if(info) cout << "Gradient Checking"<<endl;

    sf.reset();
    network.clear();
    sf.next();

    //e along
    if(info) cout<<"gc F Layer E ... ";
    flayer->delta.fill(1.0);
    flayer->fp(&ilayer->out);
    flayer->bp(ilayer);
    for(int i=0; i<flayer->Nodes; ++i){
        double &tval = flayer->node[i].expect;
        double del = 0;
        double bpdel;

        bpdel = flayer->node[i].eupdate;

        tval += delta;
        flayer->fp(&ilayer->out);
        del += flayer->out(i);

        tval -= 2*delta;
        flayer->fp(&ilayer->out);
        del -= flayer->out(i);

        tval += delta;

        del /= 2 * delta;

        //cout<< "gc fLayer e("<< i<< ") bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            if(info) cout<<" fail "<<bpdel*100/del<<"%"<<endl;
            return false;
        }else{
        }
    }
    if(info) cout<<" success"<<endl;

    //v along
    if(info) cout<<"gc F Layer V ... ";
    flayer->delta.fill(1.0);
    flayer->fp(&ilayer->out);
    flayer->bp(ilayer);
    for(int i=0; i<flayer->Nodes; ++i){
        double &tval = flayer->node[i].variance;
        double del = 0;
        double bpdel;

        bpdel = flayer->node[i].vupdate;

        tval += delta;
        flayer->fp(&ilayer->out);
        del += flayer->out(i);

        tval -= 2*delta;
        flayer->fp(&ilayer->out);
        del -= flayer->out(i);

        tval += delta;

        del /= 2 * delta;

        //cout<< "gc fLayer v("<< i<< ") bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            if(info) cout<<" fail "<<bpdel*100/del<<"%"<<endl;
            return false;
        }else{
        }
    }
    if(info) cout<<" success"<<endl;

    //player along

    if(info) cout<<"gc P Layer ... ";
    player->delta.fill(1.0);
    player->fp(&flayer->out);
    player->bp(flayer);
    for(int i=0; i<flayer->Nodes; ++i){
        double &tval = flayer->out(i);
        double del = 0;
        double bpdel;


        bpdel = flayer->delta(i);

        tval += delta;
        player->fp(&flayer->out);
        del += accu(player->out);

        tval -= 2*delta;
        player->fp(&flayer->out);
        del -= accu(player->out);

        tval += delta;

        del /= 2 * delta;

        //cout<< "gc pLayer delta("<< i<< ") bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            if(info) cout<<"fail "<<bpdel*100/del<<"%"<<endl;
            return false;
        }else{
        }
    }
    if(info) cout<<"success"<<endl;

    //nlayer along
    if(info) cout<<"gc N Layer ... ";
    nlayer->delta.fill(1.0);
    nlayer->fp(&player->out);
    nlayer->bp(player);
    for(int i=0; i<player->Nodes; ++i){
        double &tval = player->out(i);
        double del = 0;
        double bpdel;

        bpdel = player->delta(i);

        tval += delta;
        nlayer->fp(&player->out);
        del += nlayer->out(i);

        tval -= 2*delta;
        nlayer->fp(&player->out);
        del -= nlayer->out(i);

        tval += delta;

        del /= 2 * delta;

        //cout<< "gc nLayer delta("<< i<< ") bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            if(info) cout<<" fail "<<bpdel*100/del<<"%"<<endl;
            return false;
        }else{
        }
    }
    if(info) cout<<" success"<<endl;

    //clayer along
    if(info) cout<<"gc C Layer ... ";
    clayer->delta.fill(1.0);
    clayer->fp(&nlayer->out);
    clayer->bp(nlayer);
    for(int i=0; i<nlayer->Nodes; ++i){
        double &tval = nlayer->out(i);
        double del = 0;
        double bpdel;


        bpdel = nlayer->delta(i);

        tval += delta;
        clayer->fp(&nlayer->out);
        del += clayer->out(i);

        tval -= 2*delta;
        clayer->fp(&nlayer->out);
        del -= clayer->out(i);

        tval += delta;

        del /= 2 * delta;

        //cout<< "gc cLayer delta("<< i<< ") bp="<< setprecision(5) <<bpdel<<" del="<< del;
        if( fabs(bpdel - del) > errorMax){
            if(info) cout<<" fail "<<bpdel*100/del<<"%"<<endl;
            return false;
        }else{
        }
    }
    if(info) cout<<" success"<<endl;

    if(info) cout << "gradient check success "<<endl;
    return true;
}
}
