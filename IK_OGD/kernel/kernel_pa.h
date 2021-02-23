#pragma once

#include "kernel_optim.h"


namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_pa: public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_pa(const Params &param,DataSet<FeatType, LabelType> &dataset,
               LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~kernel_pa();
	float C;
protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
	virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_pa<FeatType, LabelType>::kernel_pa(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = "kernel_pa";
	this->C=param.C;
}

template <typename FeatType, typename LabelType>
kernel_pa<FeatType, LabelType>::~kernel_pa()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float kernel_pa<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    float y = this->Predict(x);

    float lt=1-x.label*y;
	//cout<<lt<<"\t";
	if(C<lt)
		lt=C;

    if(lt>0)
    {
        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*lt,x);
        add_SV(support);
    }
    return y;
}


template <typename FeatType, typename LabelType>
float kernel_pa<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
{
    float predict = 0;

    SV<FeatType, LabelType>* p_predict = this->SV_begin;
    while (p_predict!=NULL)
    {
        predict+=p_predict->SV_alpha* kern(p_predict->SV_data,data);
        p_predict=p_predict->next;
    }
    return predict;
}



}
