
#pragma once

#include "kernel_optim.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_RBP: public Kernel_optim<FeatType, LabelType>
{

protected:
    int Budget;
public:
    kernel_RBP(const Params &param,DataSet<FeatType, LabelType> &dataset,
               LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~ kernel_RBP();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
			virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_RBP<FeatType, LabelType>:: kernel_RBP(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = " kernel_RBP";
    this->Budget=param.Budget_set;
}

template <typename FeatType, typename LabelType>
kernel_RBP<FeatType, LabelType>::~ kernel_RBP()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float  kernel_RBP<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    float y = this->Predict(x);
    if (y*x.label<=0)
    {

        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label,x);
        this->add_SV(support);

    }
    //delete SV
    if(this->size_SV==Budget+1)
    {
        srand((unsigned)time(NULL));
        int SV_to_delete=rand() % (Budget);//from 0 to Budget-1
        this->delete_SV(SV_to_delete);
    }
    return y;
}
template <typename FeatType, typename LabelType>
float kernel_RBP<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
{
    float predict = 0;

    SV<FeatType, LabelType>* p_predict = this->SV_begin;
    while (p_predict!=NULL)
    {
        predict+=p_predict->SV_alpha* this->kern(p_predict->SV_data,data);
        p_predict=p_predict->next;
    }
    return predict;
}


}
