#pragma once

#include "kernel_optim.h"
#include <iostream>

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_bpas: public Kernel_optim<FeatType, LabelType>
{

protected:
    int Budget;
    float C_bpas;
public:
    kernel_bpas(const Params &param,DataSet<FeatType, LabelType> &dataset,
                LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~ kernel_bpas();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
			virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_bpas<FeatType, LabelType>:: kernel_bpas(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,
            dataset, lossFunc)
{
    this->id_str = " kernel_BPAS";
this->Budget=param.Budget_set;
    this->C_bpas=param.C_bpas;
}

template <typename FeatType, typename LabelType>
kernel_bpas<FeatType, LabelType>::~kernel_bpas()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float  kernel_bpas<FeatType,LabelType>::UpdateWeightVec(
    const DataPoint<FeatType, LabelType> &x)
{
    float y=0;
    float *k_t=NULL;
    //calculate k_t
    if(this->size_SV!=0)
    {
        SV<FeatType, LabelType>* p_predict=this->SV_begin;
        k_t=new float [this->size_SV];
        int i=0;
        while (p_predict!=NULL)
        {
            k_t[i]=this->kern(p_predict->SV_data,x);
            p_predict=p_predict->next;
            i++;
        }

        //k_t done

        //get prediction
        p_predict=this->SV_begin;
        i=0;
        while (p_predict!=NULL)
        {
            y+=p_predict->SV_alpha* k_t[i];
            p_predict=p_predict->next;
            i++;
        }
    }
    //prediction is in y
    float l_t=1-x.label*y;
    if(l_t<0)
    {
        l_t=0;
    }

    //get the Hinge Loss

    if (l_t>0)
    {
        float tao= (std::min)(C_bpas,l_t);
        if(this->size_SV<Budget)
        {
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*tao,x);
            this->add_SV(support);
        }
        else  //full Budget
        {
            double Q_star=1000000;
            int star=1;
            double star_alpha=1.0;

            SV<FeatType, LabelType> *p_search=this->SV_begin;

            for(int i=0; i<this->size_SV; i++)
            {
                double k_rt=k_t[i];
                double alpha_r=p_search->SV_alpha;
                double beta_t=alpha_r*k_rt+tao*x.label;
                double distance=alpha_r*alpha_r+beta_t*beta_t-2*beta_t*alpha_r*k_rt;
                double f_rt=y-alpha_r*k_rt+beta_t;
                double l_rt=1-x.label*f_rt;
                if(l_rt<0)
                    l_rt=0;
                double Q_r=0.5*distance+C_bpas*l_rt;
                if(Q_r<Q_star)
                {
                    Q_star=Q_r;
                    star=i;
                    star_alpha= beta_t;
                }
                p_search=p_search->next;
            }
            this->delete_SV(star);
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(float(star_alpha),x);
            this->add_SV(support);
        }
    }
	delete [] k_t;
    return y;
}

template <typename FeatType, typename LabelType>
float kernel_bpas<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
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
