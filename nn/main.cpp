#include "ANNModel.hpp"
#include "output/include/plot.hpp"
#include <string>
#include <iostream>
#include <chrono>
#include "Timer.hpp"
#include "layer/include/anfis.hpp"
#include "output/include/Info.hpp"

using namespace std;

int main(int argc, char* argv[]){
    //anfis test
    auto f = cout.flags();
    nn::ANFISModel anfis;
    //anfis.load("test/anfis_t0.nn");
    anfis.load("test/anfis_t1.nn");
    anfis.GradientCheck(true);
    //return 0;

    //auto anfis = nn::anfis::CreateAnfis_Type3(2,2,1);
    nn::showNetwork(anfis.network);
    anfis.trainer.train();
    cout.flags(f);
    anfis.tester.test();

    return 0;
    /*
    */

    if(argc != 2){
        cout<<"data" << endl;
        return -1;
    }

    string path(argv[1]);

    nn::ANNModel nnm;

    TIMER_MEASURE_MACRO( if(!nnm.load(path)) return -1; , "Load : " );

    TIMER_MEASURE_MACRO( if(!nnm.tester.gradientChecking(true))return -1; ,"GradientCheck : ");

    TIMER_MEASURE_MACRO( nnm.trainer.train(); , "Train : ");

    TIMER_MEASURE_MACRO( nnm.tester.test(nn::Tester::classification); , "Test : ");

    //drawResult2D(nnm);

    return 0;
}
