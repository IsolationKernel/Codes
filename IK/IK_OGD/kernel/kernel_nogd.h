

#pragma once

#include "kernel_optim.h"
#include <eigen3/Eigen/Eigen>
#include "cmath"
using namespace Eigen;

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_nogd: public Kernel_optim<FeatType, LabelType>
{

protected:
    int k_nogd;
    MatrixXf * K_budget;
    virtual ~kernel_nogd();
    int Budget;
    VectorXf *w_nogd;
	VectorXf *w_nogd_sum;
    MatrixXf * M_nogd;
    bool flag;
	int num_update;
	float eta1;

public:
    kernel_nogd(const Params &param,DataSet<FeatType, LabelType> &dataset,
                LossFunction<FeatType, LabelType> &lossFunc);

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	virtual float Predict(const DataPoint<FeatType, LabelType> &data);
	virtual void begin_test(void);
};

template <typename FeatType, typename LabelType>
kernel_nogd<FeatType, LabelType>::kernel_nogd(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc ): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
	 eta1=param.eta1;
	 this->eta0=param.eta;
	num_update=0;
    this->id_str = "kernel_nogd";
    this->k_nogd=param.k_nogd;
    this->Budget=param.Budget_set;
    this->K_budget=new MatrixXf(Budget,Budget);
    for(int i=0; i<Budget; i++)
    {
        (*K_budget)(i,i)=1;
    }
    this->w_nogd=new VectorXf(k_nogd);
    for(int i=0; i<k_nogd; i++)
    {
        (*w_nogd)(i)=0;
    }
	
	this->w_nogd_sum=new VectorXf(k_nogd);
    for(int i=0; i<k_nogd; i++)
    {
        (*w_nogd_sum)(i)=0;
    }

    this->M_nogd= new MatrixXf(k_nogd,Budget);
    this->flag=0;
}

template <typename FeatType, typename LabelType>
kernel_nogd<FeatType, LabelType>::~kernel_nogd()
{
    delete w_nogd;
    delete M_nogd;
    delete K_budget;
}

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float kernel_nogd<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    float y=0;
    VectorXf kt(this->size_SV);
    VectorXf zt(k_nogd);
    //calculate k_t
    if((this->size_SV!=0)&&(flag==0))
    {
        SV<FeatType, LabelType>* p_predict=this->SV_begin;
        int i=0;
        while (p_predict!=NULL)
        {
            kt(i)=this->kern(p_predict->SV_data,x);
            p_predict=p_predict->next;
            i++;
        }
        //k_t done

        //get prediction
        p_predict=this->SV_begin;
        i=0;
        while (p_predict!=NULL)
        {
            y+=p_predict->SV_alpha* kt(i);
            p_predict=p_predict->next;
            i++;
        }
    }
    if(flag!=0) //linear predict
    {
        SV<FeatType, LabelType>* p_predict=this->SV_begin;
        int i=0;
        while (p_predict!=NULL)
        {
            kt[i]=this->kern(p_predict->SV_data,x);
            p_predict=p_predict->next;
            i++;
        }
        zt=(*M_nogd)*kt;
        y=(*w_nogd).dot(zt);
    }
    //update
    if(y*x.label<1)
    {
        if(this->size_SV<Budget) //kernel update
        {
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*this->eta0,x);
            this->add_SV(support);

            for(int i=0; i<this->size_SV-1; i++)
            {
                (*K_budget)(i,this->size_SV-1)=kt(i);
                (*K_budget)(this->size_SV-1,i)=kt(i);
            }

        }
        else
        {
            if(flag==0) //SVD
            {
				this->curIterNum=1;
                flag=1;
                EigenSolver<MatrixXf> es(*K_budget);
                MatrixXcf V = es.eigenvectors();
                //cout<<es.eigenvalues()<<endl;

                for(int i=0; i<k_nogd; i++)
                {
                    float length=0;
                    for(int j=0; j<Budget; j++)
                    {
                        length=length+V(j,i).real()*V(j,i).real();
                    }
                    for(int j=0; j<Budget; j++)
                    {
                        V(j,i)=V(j,i)/length;
                    }
                }

                for(int i=0; i<k_nogd; i++)
                {
                    for(int j=0; j<Budget; j++)
                    {
                        (*M_nogd)(i,j)=V(j,i).real()/sqrt(es.eigenvalues()[i].real());
                    }
                }
                zt=(*M_nogd)*kt;
                (*w_nogd)=(*w_nogd)+eta1*x.label*zt;
				(*w_nogd_sum)=(*w_nogd_sum)+(*w_nogd);
				num_update++;
            }
            else
            {
                (*w_nogd)=(*w_nogd)+eta1*x.label*zt;
				(*w_nogd_sum)=(*w_nogd_sum)+(*w_nogd);
				num_update++;
            }
        }
    }

    return y;
}

template <typename FeatType, typename LabelType>
void kernel_nogd<FeatType, LabelType>::begin_test(void)
{
	(*w_nogd)=(*w_nogd_sum)/float(num_update);
}

template <typename FeatType, typename LabelType>
float kernel_nogd<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
{
       float y=0;
       SV<FeatType, LabelType>* p_predict=this->SV_begin;
        int i=0;
		VectorXf kt(this->size_SV);
        VectorXf zt(k_nogd);
        while (p_predict!=NULL)
        {
            kt[i]=this->kern(p_predict->SV_data,data);
            p_predict=p_predict->next;
            i++;
        }
        zt=(*M_nogd)*kt;
        y=(*w_nogd).dot(zt);
		return y;
}



}
