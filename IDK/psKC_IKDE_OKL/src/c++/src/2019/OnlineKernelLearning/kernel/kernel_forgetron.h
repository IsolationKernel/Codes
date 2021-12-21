
#pragma once

#include "kernel_optim.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "../data/DataPoint.h"
#include <math.h>
#include <cmath>

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_forgetron: public Kernel_optim<FeatType, LabelType>
{
protected:
    int Budget;
    int err_until_now;
    double Q;

public:
    kernel_forgetron(const Params &param,DataSet<FeatType, LabelType> &dataset,
                     LossFunction<FeatType, LabelType> &lossFunc);

    virtual ~kernel_forgetron();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
			virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_forgetron<FeatType, LabelType>:: kernel_forgetron(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = " kernel_forgetron";
this->Budget=param.Budget_set;
    this->err_until_now=0;
    this->Q=0;
}

template <typename FeatType, typename LabelType>
kernel_forgetron<FeatType, LabelType>::~ kernel_forgetron()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float  kernel_forgetron<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    float y = this->Predict(x);
    if (y*x.label<=0)
    {
        err_until_now++;

        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label,x);
        this->add_SV(support);
    }

    //delete SV
    if(this->size_SV==Budget+1)
    {
        float predict = this->Predict(this->SV_begin->SV_data);

        double mu=this->SV_begin->SV_data.label*predict;
        double delta=this->SV_begin->SV_alpha/this->SV_begin->SV_data.label;

        double coeA=delta*delta-2*delta*mu;
        double coeB=2*delta;
        double coeC=Q-(15.0/32.0)*err_until_now;

        double phi=0;
        if (coeA==0)
            phi=(std::max)(0.0,(std::min)(1.0,-coeC/coeB));
        else if (coeA>0)
        {
            if (coeA+coeB+coeC<=0)
                phi=1;
            else
                phi=(-coeB+sqrt(coeB*coeB-4*coeA*coeC))/(2*coeA);
        }
        else if (coeA<0)
        {
            if (coeA+coeB+coeC<=0)
                phi=1;
            else
                phi=(-coeB-sqrt(coeB*coeB-4*coeA*coeC))/(2*coeA);
        }

        //alpha=phi*alpha_t;
        SV<FeatType, LabelType>* p_change_alpha=this->SV_begin;
        while(p_change_alpha!=NULL)
        {
            p_change_alpha->SV_alpha= (float)(p_change_alpha->SV_alpha*phi);
            p_change_alpha=p_change_alpha->next;
        }

        Q=Q+(delta*phi)*(delta*phi)+2*delta*phi*(1-phi*mu);
        this->delete_SV(0);
    }
    return y;
}
template <typename FeatType, typename LabelType>
float kernel_forgetron<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
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
