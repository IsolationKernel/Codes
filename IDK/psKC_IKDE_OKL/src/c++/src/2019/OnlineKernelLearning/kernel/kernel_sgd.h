#pragma once

#include "kernel_optim.h"


namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_sgd: public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_sgd(const Params &param,DataSet<FeatType, LabelType> &dataset,
               LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~kernel_sgd();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
	virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_sgd<FeatType, LabelType>::kernel_sgd(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = "kernel_ogd";
	this->eta0=param.eta;
}

template <typename FeatType, typename LabelType>
kernel_sgd<FeatType, LabelType>::~kernel_sgd()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float kernel_sgd<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    float y = this->Predict(x);

    float gt_i = this->lossFunc->GetGradient(x.label,y);

    if(gt_i!=0)
    {
        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(-this->eta0 * gt_i,x);
        this->add_SV(support);
    }
    return y;
}


template <typename FeatType, typename LabelType>
float kernel_sgd<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
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
