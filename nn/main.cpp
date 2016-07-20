#include "ANNModel.hpp"

#include <string>
#include <iostream>

using namespace std;

int main(int argc, char* argv[]){

    if(argc != 2){
        cout <<"data" << endl;
        return -1;
    }

    string path(argv[1]);

    nn::ANNModel nnm;

    if(!nnm.load(path)) return -1;

    //if(!nn::gradientChecking()) return -1;

    nnm.trainer.train();

    return 0;
}
