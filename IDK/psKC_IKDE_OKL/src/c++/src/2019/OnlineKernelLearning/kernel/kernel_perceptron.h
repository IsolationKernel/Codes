

#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_perceptron: public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_perceptron(const Params &param,DataSet<FeatType, LabelType> &dataset,
                      LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~ kernel_perceptron();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
			virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_perceptron<FeatType, LabelType>:: kernel_perceptron(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = " kernel_perceptron";
}

template <typename FeatType, typename LabelType>
kernel_perceptron<FeatType, LabelType>::~ kernel_perceptron()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float  kernel_perceptron<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    float y = this->Predict(x);

    if (y*x.label<=0)
    {
        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label,x);

        this->add_SV(support);
    }
    return y;
}
template <typename FeatType, typename LabelType>
float kernel_perceptron<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
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
