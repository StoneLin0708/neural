#include "ANNModel.hpp"
//#include "core/include/Layer.hpp"
#include <string>
#include <iostream>
#include <omp.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace nn;

int main(int argc, char* argv[]){

    if(argc != 2){
        cout <<"data" << endl;
        return -1;
    }

    string path(argv[1]);

    nn::ANNModel nnm;

    if(!nnm.load(path)) return -1;

    //if(!nn::gradientChecking()) return -1;

    nnm.trainer.gradientChecking();
    nnm.trainer.train();

    cv::Mat m(500,500,CV_8UC3,Scalar(255,255,255));

    for(int y=0;y<500; ++y){
        for(int x=0;x<500; ++x){
            static_cast<InputLayer*>(nnm.network.Layer[0])->out(0) = x/500.0;
            static_cast<InputLayer*>(nnm.network.Layer[0])->out(1) = y/500.0;
            nnm.network.fp();
            /*
            if(y%100==0&&x%100==0){
                cout <<"i"<<nnm.network.Layer[0]->out;
                cout <<"o"<<nnm.network.Layer.back()->out<<endl;
                cin.get();
            }
            */
            /*
            if(	static_cast<OutputLayer*>(nnm.network.Layer.back())->out(0)>
                static_cast<OutputLayer*>(nnm.network.Layer.back())->out(1)){
                */
            if(	static_cast<OutputLayer*>(nnm.network.Layer.back())->out(0)> 0.5){
                m.at<Vec3b>(y,x) = Vec3b(255,0,0);
            }
            else
                m.at<Vec3b>(y,x) = Vec3b(0,255,0);
        }
    }
    imshow("m",m);
    waitKey(0);

    return 0;
}
