#include "neural.hpp"
#include "stringCheck.hpp"
#include "nnfun.hpp"
#include "nnio.hpp"
#include "sampleSet.hpp"
#include <unistd.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace arma;

nn::nn(nnParam param){
	_init = true;
	_param = param;
	//readnn(path, _param);
	if(!enableParam())
		_init = false;
	show();
	showParam();
}

inline
void nn::test(){
	//input to hidden 0
	Lhidden[0].sum = Linput.out * Lhidden[0].weight;
	Lhidden[0].act(Lhidden[0].sum, Lhidden[0].out, Lhidden[0].n_node()-1);
	//hidden 1 to hidden n
	for(int i=1; i<(int)Lhidden.size(); ++i){
		Lhidden[i].sum = Lhidden[i-1].out * Lhidden[i].weight;
		Lhidden[i].act(Lhidden[i].sum, Lhidden[i].out, Lhidden[i].n_node()-1);
	}
	//hidden n to output
	Loutput.sum = Lhidden.back().out * Loutput.weight;
	Loutput.act(Loutput.sum, Loutput.out, Loutput.n_node());
}

inline void nn::clear_wupdates(){
	Loutput.cost.zeros();
	Loutput.costnmse.zeros();
	for(int i=0; i<(int)Lhidden.size(); ++i)
		Lhidden[i].wupdates.zeros();
	Loutput.wupdates.zeros();
}

inline void nn::bp(){
	//output to last hidden
	const int nodeso = Loutput.n_node();
	//compute delta
	Loutput.dact( Loutput.sum , Loutput.delta, nodeso);
	for(int i=0; i<nodeso; ++i)
		Loutput.delta(i) *= dcost(Loutput.desireOut(i), Loutput.out(i));
	//compute cost
	for(int i=0; i<nodeso; ++i)
		Loutput.cost(i) += cost(Loutput.desireOut(i), Loutput.out(i));
	for(int i=0; i<nodeso; ++i)
		Loutput.costnmse(i) += nn_func::nmse(Loutput.desireOut(i), Loutput.out(i));
	//compute wupdate
	const int nodeht = Lhidden.back().n_node();
	for(int i=0; i<nodeso; ++i)
		for(int j=0; j<nodeht; ++j)
			Loutput.wupdate(j,i) = learningRate * Loutput.delta(i) * Lhidden.back().out(j);
	Loutput.wupdates -= Loutput.wupdate;
	//hidden to hidden || hidden to output
	mat *weightU = &Loutput.weight; //weight upper layer
	rowvec *deltaU = &Loutput.delta; //delta upper layer
	int nodesU = nodeso;
	int nodesh;
	int inputsh;
	rowvec *outD = NULL; //output lower layer
	for(int layer=Lhidden.size()-1; layer>=0; --layer){
		Lhidden[layer].delta.zeros(); //reset delta
		if(layer == 0)
			outD = &Linput.out; //hidden 1 to input
		else
			outD = &Lhidden[layer-1].out; //hidden layer to layer-1

		nodesh = Lhidden[layer].n_node();
		for(int i=0; i<nodesh; ++i){
			//compute summmation of upper layer to this layer delta
			for(int j=0; j<nodesU; ++j)
				Lhidden[layer].delta(i) += (*deltaU)(j) * (*weightU)(i,j);
		}
		Lhidden[layer].dact(Lhidden[layer].sum, Lhidden[layer].sum, nodesh);

		inputsh = Lhidden[layer].n_input();
		for(int i=0; i<nodesh; ++i){
			//compute delta
			Lhidden[layer].delta(i) *= Lhidden[layer].sum(i);
			//compute weight
			for(int j=0; j<inputsh; ++j)
				Lhidden[layer].wupdate(j,i) = learningRate * Lhidden[layer].delta(i) * (*outD)(j);
		}
		Lhidden[layer].wupdates -= Lhidden[layer].wupdate;

		weightU = &(Lhidden[layer].weight);
		deltaU = &(Lhidden[layer].delta);
		nodesU = nodesh;
	}
}

bool nn::gradientChecking(int sample){
	if(sample == -1)
		sample = _n_sample;
	double delta = 1.0e-4;
	double errorMax = 1.0e-6;
	double fw,fwd;
	int nodes = Loutput.n_node();
	int inputs = Loutput.n_input();
	mat *weight = &Loutput.weight;
	mat *wupdates = &Loutput.wupdates;
	int layers = 1+Lhidden.size();

	return true;
	for(int l=layers-1; l>=0; --l){
		if(l == layers-1) cout << "gradient check layer output" << endl;
		else cout << "gradient check layer hidden [" << l+1 <<']'<<endl;
		//compute delta weight
		clear_wupdates();
		for(int s=0; s<sample; ++s){
			Linput.setFeatures(s);
			Loutput.setOutput(s);
			test(); //forward
			bp(); //backpropagation
		}
		for(int i=0; i<nodes; ++i){
			for(int j=0; j<inputs; ++j){
				//because wupdates include learning rate so divide it
				fw = -1*( (*wupdates)(j,i)/learningRate)/sample;

				//w = w + h
				(*weight)(j,i) += delta;
				fwd = 0;
				for(int s=0; s<sample; ++s){
					Linput.setFeatures(s);
					Loutput.setOutput(s);
					test();
					//f(w+h) summation cost
					for(int k=0; k<Loutput.n_node(); ++k)
						fwd += cost(Loutput.desireOut(k), Loutput.out(k));
				}
				//set w = w - h (w = w + h in last step ,- 2*h)
				(*weight)(j,i) -= 2*delta;
				for(int s=0; s<sample; ++s){
					Linput.setFeatures(s);
					Loutput.setOutput(s);
					test();
					//f(w+h) - f(w-h) summation cost
					for(int k=0; k<Loutput.n_node(); ++k)
						fwd -= cost(Loutput.desireOut(k), Loutput.out(k));
				}
				//reset w to w
				(*weight)(j,i) += delta;
				fwd /= 2*delta*sample;
				//cout << j << ',' << i << ",fw="<<fw<<",fwd="<< fwd<<endl;
				if( abs(fwd - fw) > errorMax ){
					/*
					if(l==layers-1)
						cout<< "gradient check fail at layer output weight("<<j<<","<<i<<")"
							<<" fw="<< fw <<" fwd="<< fwd
							<< " fw/fwd=" << fw/fwd <<endl;
					else
						cout<< "gradient check fail at layer hidden["<< l+1 <<"] weight("<<j<<","<<i<<")"
							<<" fw="<< fw <<" fwd="<< fwd
							<< " fw/fwd=" << fw/fwd <<endl;
							*/
					//return false;
				}
			}
		}
		if(l>0){
			nodes = Lhidden[l-1].n_node()-1;
			inputs = Lhidden[l-1].n_input();
			weight = &Lhidden[l-1].weight;
			wupdates = &Lhidden[l-1].wupdates;
		}
	}

	cout << "gradient check success "<<endl;
	return true;
}

inline
void nn::wupdate(){
	static int trainNumber = _param.trainNumber;
	Loutput.weight *= 0.999;
	Loutput.weight += Loutput.wupdates/trainNumber;
	for(int i=0; i<(int)Lhidden.size(); ++i){
		Lhidden[i].weight *= 0.999;
		Lhidden[i].weight += Lhidden[i].wupdates/trainNumber;
	}
}

void nn::error(int &i){
	static int ep = ceil((double)iteration/1000);
	static double last_t = clock();
	static double last_c = 0;
	static int stop_counter = 0;
	static const double startTime = clock();

	double errs = 0;
	double nmse = 0;

	bool show = ( ( clock() - last_t ) >= 500000);
	bool save = ( i%ep == 0 );
	if(show || save){
		for(int j=0; j<Loutput.n_node(); ++j){
			errs += Loutput.cost(j)/_param.trainNumber/Loutput.n_node();
			nmse += Loutput.costnmse(j)/_param.trainNumber/Loutput.n_node();
		}
		if( abs(errs - last_c) < _param.stopTrainingCost)
			stop_counter++;
		else
			stop_counter =0;

		if(stop_counter==10)
		{
			i = iteration;
			cout << endl <<" cost limit , stop training "<<endl;
		}
	}
	if( show ){
		cout.flush();
		int spendTime = (int)((clock()-startTime)/CLOCKS_PER_SEC);
		cout<< /*'\r'<<*/ " iteration : " <<  setw(7) << i
			<<" average cost : " << setw(7) << errs
			<<" nmse : " << setw(7) << nmse
			<<" cost rate : " << setw(7) << abs(errs-last_c)
			<<" , " << setw(7) << (errs-last_c)*100/errs << "% "
			<<" spend time " << spendTime
			<<"s left " << (int)( (iteration-i) * spendTime / i)<<"s   "<<endl;

		last_t = clock();
		last_c = errs;
	}
	if( save ){
		e.push_back(errs);
		en.push_back(nmse);
	}
}

void nn::train(){
	/*
	for(int i=0; i<_n_sample; ++i)
		cout << outputs(i) << " : " << features[i];
		*/
	sampleSet::param param = {_n_sample,_param.trainStart,_param.trainEnd,_param.trainNumber};
	sampleSet sampleSet( _param.trainType ,param);

	int s;
	for(int i=0; i<iteration; ++i){
	//for(int i=0; i<2; ++i){
		clear_wupdates();
		while(!sampleSet.last()){
			s = sampleSet.getNext();
	/*
	cout<< "-----------iteration" << i << "-----------" << endl;
			cout << "sample : " << s<<endl;
			cin.get();
	cout<< "-----------forword--------------" << endl;
	*/
			Linput.setFeatures(s);
			test();
	//cout<< "-----------bp-------------------" << endl;
			Loutput.setOutput(s);
			bp();
/*
			cout <<"s=" << s <<Linput.out << Loutput.desireOut << endl;
			cin.get();
			*/
		}
		error(i);
		wupdate();
		//if(i%1000 ==1000)
			//testResult();
	}
	/*
	cout << Loutput.weight << endl;
	cout << Lhidden[0].weight << endl;
	*/
	cout << endl;
}

