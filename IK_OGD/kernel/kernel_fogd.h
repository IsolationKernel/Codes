#pragma once
#include <iostream>
#include <random>
#include "kernel_optim.h"
#include <math.h>
#include <time.h>

#define B_cos 1.273239
#define P_cos 0.225
#define C_cos -0.40528
#define pi_cos 3.1415926

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_fogd: public Kernel_optim<FeatType, LabelType>
{

protected:
    int D;
    IndexType u_dimension;
    s_array<double> w_fogd;
    s_array<double> u;
    s_array<double> w_fogd_sum;
	s_array<double> ux;
    s_array<double> ux_cos;
	int num_update;



	double a;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

public:
    kernel_fogd(const Params &param,DataSet<FeatType, LabelType> &dataset,
                LossFunction<FeatType, LabelType> &lossFunc);

    virtual ~kernel_fogd();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
		virtual void begin_test(void);
};

template <typename FeatType, typename LabelType>
kernel_fogd<FeatType, LabelType>::kernel_fogd(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = "kernel_fogd";
    this->D=param.D_set;

    w_fogd.resize(2 * D);
    w_fogd.zeros();
	w_fogd_sum.resize(2*D);
	w_fogd_sum.zeros();
    this->ux.resize(D);
    this->ux_cos.resize(2 * D);
	num_update=0;

    this->u_dimension=0;
    this->distribution=normal_distribution<double>(0.0,sqrt(param.gamma*2));
    this->generator=default_random_engine((unsigned)time(NULL));
    this->eta0 = param.eta;
}

template <typename FeatType, typename LabelType>
kernel_fogd<FeatType, LabelType>::~kernel_fogd()
{
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float kernel_fogd<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
	
    IndexType x_dimension=x.max_index;
    //generate u
    if(u_dimension<x.max_index)
    {
        //update dimension
       this->u.reserve(D * x_dimension);
       this->u.resize(D * x_dimension);
       for(IndexType i=(D*u_dimension); i<(D*x_dimension); i++)
            this->u[i]=distribution(generator);
       this->u_dimension=x_dimension;
    }

    this->ux.zeros();

    size_t index_begin;
    float feature;
    for(size_t j=0; j<x.indexes.size(); j++)
    {
        index_begin=(x.indexes[j]-1)*D;
        feature=x.features[j];
        for(int i=0; i<D; i++)
        {
            ux[i]+=u[index_begin]*feature;
            index_begin++;
        }
    }
	double *p1=ux_cos.begin;
	double *p2=p1+D;

    for(int i=0; i<D; i++)
    {
		while(ux[i]< -3.14159265)
			ux[i]+= 6.28318531;
		while(ux[i]> 3.14159265)
			ux[i]-= 6.28318531;
		a = B_cos * ux[i] + C_cos * ux[i] * abs(ux[i]);
        *p1 = P_cos * (a * abs(a) - a) + a;

        *p2=sqrt(1-(*p1)*(*p1));
		if(ux[i]<0)
			*p2=-(*p2);

		p1++;
		p2++;
	}

    double y=0;
    for(int i=0; i<2*D; i++)
        y=y+w_fogd[i]*ux_cos[i];

    if(y*x.label<1)
    {
		num_update++;
        for(int i=0; i<2*D; i++)
        {   
			w_fogd[i]=w_fogd[i]+this->eta0*x.label*ux_cos[i];
		    w_fogd_sum[i]=w_fogd_sum[i]+w_fogd[i];		
		}
    }
    return float(y);
}

template <typename FeatType, typename LabelType>
float kernel_fogd<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
{

    this->ux.zeros();

    size_t index_begin;
    float feature;
    for(size_t j=0; j<data.indexes.size(); j++)
    {
        index_begin=(data.indexes[j]-1)*D;
        feature=data.features[j];
        for(int i=0; i<D; i++)
        {
            ux[i]+=u[index_begin]*feature;
            index_begin++;
        }
    }
	double *p1=ux_cos.begin;
	double *p2=p1+D;

    for(int i=0; i<D; i++)
    {
		while(ux[i]< -3.14159265)
			ux[i]+= 6.28318531;
		while(ux[i]> 3.14159265)
			ux[i]-= 6.28318531;
		a = B_cos * ux[i] + C_cos * ux[i] * abs(ux[i]);
        *p1 = P_cos * (a * abs(a) - a) + a;
        *p2=sqrt(1-(*p1)*(*p1));
		if(ux[i]<0)
			*p2=-(*p2);
		p1++;
		p2++;
	}

    double y=0;
    for(int i=0; i<2*D; i++)
        y=y+w_fogd[i]*ux_cos[i];
    return float(y);
}

template <typename FeatType, typename LabelType>
void kernel_fogd<FeatType, LabelType>::begin_test(void)
{
        for(int i=0; i<2*D; i++)
        {   
		    w_fogd[i]=w_fogd_sum[i]/num_update;	
		}
}

}
