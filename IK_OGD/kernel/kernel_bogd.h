

#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_bogd: public Kernel_optim<FeatType, LabelType>
{

protected:
    int Budget;
    float lambda;

public:
    kernel_bogd(const Params &param,DataSet<FeatType, LabelType> &dataset,
                LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~kernel_bogd();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
			virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_bogd<FeatType, LabelType>::kernel_bogd(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = "kernel_bogd";
    this->Budget=param.Budget_set;
    this->lambda=param.lambda;
	this->eta0=param.eta;
}

template <typename FeatType, typename LabelType>
kernel_bogd<FeatType, LabelType>::~kernel_bogd()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float kernel_bogd<FeatType,LabelType>::UpdateWeightVec(const
        DataPoint<FeatType, LabelType> &x)
{
    float y = this->Predict(x);

    float gt_i = this->lossFunc->GetGradient(x.label,y);

    SV<FeatType, LabelType>* p_alpha=this->SV_begin;
    while(p_alpha!=NULL)
    {
        p_alpha->SV_alpha=p_alpha->SV_alpha*(1-this->eta0*lambda);
        p_alpha=p_alpha->next;
    }
    if(gt_i!=0)
    {
        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(-this->eta0 * gt_i,x);
        this->add_SV(support);
    }
    //delete SV
    if(this->size_SV==Budget+1)
        this->delete_SV();

    return y;
}

template <typename FeatType, typename LabelType>
float kernel_bogd<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
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
