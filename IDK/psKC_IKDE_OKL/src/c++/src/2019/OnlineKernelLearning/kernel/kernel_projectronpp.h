#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_projectronpp: public Kernel_optim<FeatType, LabelType>
{

protected:
    int Budget;
    float U;
    s_array<float> K_inverse;
    s_array<float> K_t;

public:
    kernel_projectronpp(const Params &param,DataSet<FeatType, LabelType> &dataset,
                        LossFunction<FeatType, LabelType> &lossFunc);
    virtual ~kernel_projectronpp();

protected:
    //this is the core of different updating algorithms
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	    virtual float Predict(const DataPoint<FeatType, LabelType> &data);
			virtual void begin_test(void){}
};

template <typename FeatType, typename LabelType>
kernel_projectronpp<FeatType, LabelType>:: kernel_projectronpp(const Params &param,
    DataSet<FeatType, LabelType> &dataset,
    LossFunction<FeatType, LabelType> &lossFunc): Kernel_optim<FeatType, LabelType>(param,dataset, lossFunc)
{
    this->id_str = " kernel_projectronpp";
this->Budget=param.Budget_set;
    this->U=(1.f/4.f)*sqrtf((Budget+1.f)/logf(Budget+1.f));

    this->K_inverse.resize(Budget*Budget);
    this->K_inverse.zeros();

    this->K_t.resize(Budget * Budget);
    this->K_t.zeros();
}

template <typename FeatType, typename LabelType>
kernel_projectronpp<FeatType, LabelType>::~ kernel_projectronpp()
{
}


//update weight vector with stochastic gradient descent
template <typename FeatType, typename LabelType>
float  kernel_projectronpp<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
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

    // if there is mistake, make update
    if(this->size_SV==0)
    {

        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label,x);

        this->add_SV(support);

        //ini K_inverse
        K_inverse[0]=1;
        K_t[0]=1;
    }
    else  //have SV
    {
        float l_t=1-x.label*y;
        if(y*x.label<=0)
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
            float k_t_d_star=0;
            for(int i=0; i<this->size_SV; i++)
            {
                k_t_d_star=k_t_d_star+k_t[i]*d_star[i];
            }
            float delta_project=1-k_t_d_star;


            //full budget projectron
            if(this->size_SV==Budget)
            {
                SV<FeatType, LabelType> *p_predict=this->SV_begin;
                for(int i=0; i<Budget; i++)
                {
                    p_predict->SV_alpha=p_predict->SV_alpha+x.label*d_star[i];
                    p_predict=p_predict->next;
                }
            }
            else  // not full
            {

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

                //updata K_t
                for(int i=0; i<this->size_SV-1; i++)
                {
                    K_t[i*Budget+this->size_SV-1]=k_t[i];
                    K_t[(this->size_SV-1)*Budget+i]=k_t[i];
                }
                K_t[(this->size_SV-1)*Budget+(this->size_SV-1)]=1;///////////////////////
            }
            delete[] d_star;
        }//mistake
        else if((l_t<1)&&(l_t>0))
        {

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
            float k_t_d_star=0;
            for(int i=0; i<this->size_SV; i++)
            {
                k_t_d_star=k_t_d_star+k_t[i]*d_star[i];
            }
            float delta_project=1-k_t_d_star;

            float power_p_k_t=0;

            for(int i=0; i<this->size_SV; i++)
            {
                for(int j=0; j<this->size_SV; j++)
                {
                    power_p_k_t=power_p_k_t+K_t[i*Budget+j]*d_star[i]*d_star[j];
                }
            }

            float tau_t= (std::min)(l_t/power_p_k_t,1.f);
            float beta_t=tau_t*(2*l_t-tau_t*power_p_k_t-2*U*sqrt(delta_project));
            if(beta_t>=0)
            {
                SV<FeatType, LabelType> *p_predict=this->SV_begin;
                for(int i=0; i<this->size_SV; i++)
                {
                    p_predict->SV_alpha=p_predict->SV_alpha+tau_t*d_star[i]*x.label;
                    p_predict=p_predict->next;
                }
            }
            delete[] d_star;
        }//margin loss
    }//have SV
    delete[] k_t;

    return y;
}
template <typename FeatType, typename LabelType>
float kernel_projectronpp<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
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
