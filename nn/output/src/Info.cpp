#include "output/include/Info.hpp"
#include <iostream>
using namespace  std;

namespace nn {

void showNetwork(Network &n){
    cout<< "----------Network----------"<< endl;
    for(int i=0; i<(int)n.Layer.size(); ++i){
    cout<< " Layer "<< n.Layer[i]->Layer<< " Nodes "<< n.Layer[i]->Nodes << endl;
    cout<< "---------------------------"<< endl;
    }
}

}
