#include "core/include/nn.hpp"

#include <string>
#include <iostream>

using namespace std;


int main(int argc,char* argv[]){
    if(argc != 2){
        cout <<"data  ,result name" << endl;
		return -1;
	}

	string path = argv[1];

    nn::Network ann();

    //nn::Load( path, ann );

    //if(!nn::gradientChecking()) return -1;

    //n.train();

	return 0;
}
