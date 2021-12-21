#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_projectron: public Kernel_optim<FeatType, LabelType>
{

protected:
    s_array<float> K_inverse;
	int Budget;
public:
    kernel_projectron(const Params &param,DataSet<FeatType, LabelType> &dataset,
                      LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~ kernel_projectron();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	virtual float Predict(const DataPoint<FeatType, LabelType> &data);
	virtual void begin_test(void){};
};

template <typename FeatType, typename LabelType>
kernel_projectron<FeatType, LabelType>:: kernel_projectron(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
	this->Budget=param.Budget_set;
    this->id_str = " kernel_projectron";
    this->K_inverse.resize(Budget*Budget);
    this->K_inverse.zeros();
}

template <typename FeatType, typename LabelType>
kernel_projectron<FeatType, LabelType>::~kernel_projectron()
{
}

//add by yuewu: 2013/12/11
//Memory optimization

//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float  kernel_projectron<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
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
    // if there is mistake, make update
    if (y*x.label<=0)
    {
        if(this->size_SV==0)
        {

            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label,x);

            this->add_SV(support);

            //ini K_inverse
            K_inverse[0]=1;
        }
        else  //have SV
        {
            // calculate d_star=K_t_inver*k_t;
            float * d_star=new float [this->size_SV];
            for(int i=0; i<this->size_SV; i++)
            {
                d_star[i]=0;
                for(int j=0; j<this->size_SV; j++)
                {
                    d_star[i]=d_star[i]+K_inverse[i*Budget+j]*k_t[j];
                }
            }

            //caculate delta
            double k_t_d_star=0;
            for(int i=0; i<this->size_SV; i++)
            {
                k_t_d_star=k_t_d_star+k_t[i]*d_star[i];
            }
            double delta_project=1-k_t_d_star;
		    

            //full budget projectron
            if(this->size_SV==Budget)
            {
                SV<FeatType, LabelType> *p_predict=this->SV_begin;
                for(int i=0; i<this->size_SV; i++)
                {
                    p_predict->SV_alpha=p_predict->SV_alpha+x.label*d_star[i];
                    p_predict=p_predict->next;
                }
            }
            else  // not full
            {
                //add SV

                SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label,x);
                this->add_SV(support);


                //updata K_inverse
                for(int i=0; i<this->size_SV-1; i++)
                {
                    for(int j=0; j<this->size_SV-1; j++)
                    {
                        K_inverse[i*Budget+j]=K_inverse[i*Budget+j]+d_star[i]*d_star[j]/delta_project;
                    }
                }
                for(int i=0; i<this->size_SV-1; i++)
                {
                    K_inverse[i*Budget+this->size_SV-1]=(-1)*d_star[i]/delta_project;
                    K_inverse[(this->size_SV-1)*Budget+i]=(-1)*d_star[i]/delta_project;
                }
                K_inverse[(this->size_SV-1)*Budget+(this->size_SV-1)]=1/delta_project;
            }
            delete[] d_star;
        }
    }
    delete[] k_t;
    return y;
}


template <typename FeatType, typename LabelType>
float kernel_projectron<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
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
