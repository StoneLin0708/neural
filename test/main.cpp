#include "model/ANNModel.hpp"
#include "output/include/plot.hpp"
#include <string>
#include <iostream>
#include <chrono>
#include "output/Timer.hpp"
#include "layer/include/anfis.hpp"
#include "output/include/Info.hpp"

using namespace std;

int main(int argc, char* argv[]){
    //anfis test
    nn::ANFISModel anfis;

    TIMER_MEASURE_MACRO(anfis.load("test/anfis_t1.nn");,"Load : ")

    TIMER_MEASURE_MACRO(anfis.GradientCheck(true);,"Gradient Check : ")
    //nn::showNetwork(anfis.network);
    TIMER_MEASURE_MACRO(anfis.trainer.train();,"Train : ")

    TIMER_MEASURE_MACRO(anfis.tester.test();,"Test : ")

    return 0;

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
