

#pragma once
#include "../data/DataPoint.h"
#include "../data/DataSet.h"
#include "../loss/LossFunction.h"
#include "../common/init_param.h"
#include "../common/util.h"

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <math.h>

namespace SOL
{

/**
*  namespace: Sparse Online Learning
*/
template <typename FeatType, typename LabelType>
struct SV
{
public:
	float SV_alpha_sum;
    float SV_alpha;
    DataPoint<FeatType, LabelType> SV_data;
    SV * next;

    SV(float alpha, DataPoint<FeatType, LabelType> x)
    {
		SV_alpha_sum=0;
        SV_alpha=alpha;
        SV_data= x.clone();
        next=NULL;
    }
};


template <typename FeatType, typename LabelType> class Kernel_optim
{
    //Iteration
protected:
    //iteration number
    unsigned int curIterNum;
	float eta_now;
    //parameters
    float eta0; //learning rate
	float gamma;
	int weight;

	bool use_average_weight;
    DataSet<FeatType, LabelType> &dataSet;

    //weight vector
protected:
    SV<FeatType, LabelType> * SV_begin;
    SV<FeatType, LabelType> * SV_end;

    double time_predict{0.0};

public:
    int size_SV;

protected:
    LossFunction<FeatType, LabelType> *lossFunc;

protected:
    string id_str;

public:
    void PrintOptInfo()const
    {
        printf("--------------------------------------------------\n");
        printf("Algorithm: %s\n",this->Id_Str().c_str());
    }

public:
    Kernel_optim(const Params &param,DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc);

public:
	void SetParameter(float gamma_a=8, float eta_a = -1);

    virtual ~Kernel_optim()
    {		
		SV<FeatType, LabelType> * SV_free;
		for(int i=0;i<size_SV;i++)
		{
			SV_free=SV_begin;
			SV_begin=SV_begin->next;
			delete SV_free;
		}		
    }
    const string& Id_Str() const
    {
        return this->id_str;
    }

protected:
    //train the data
    float Train();
    //predict a new feature
	void sum_SV();
    //this is the core of different updating algorithms
    //return the predict
    virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) = 0;
	virtual float Predict(const DataPoint<FeatType, LabelType> &data) = 0;
	virtual void begin_test(void)=0;
public:

    float kern(
        const DataPoint<FeatType, LabelType> &SV_data,
        const DataPoint<FeatType, LabelType> &x);
    void add_SV(SV<FeatType, LabelType> *p_newSV);
    void delete_SV(int index_SV=0);
public:
    //learn a model
    inline float Learn(int numOfTimes = 1);
    //learn a model and return the mistake rate and its variance
    float Learn(float &aveErrRate, float &varErrRate, float &sparseRate, int numOfTimes = 1);
    //test the performance on the given set
    float Test(const Params &param, DataSet<FeatType, LabelType> &testSet);

    virtual void Online(Params const & param);
};

	template <typename FeatType, typename LabelType>
	void Kernel_optim<FeatType, LabelType>::SetParameter(float gamma_a , float eta_a) {
		this->gamma  = gamma_a;
		this->eta0 = eta_a;
	}


template <typename FeatType, typename LabelType>
Kernel_optim<FeatType, LabelType>::Kernel_optim(const Params &param,DataSet<FeatType, LabelType> &dataset,
        LossFunction<FeatType, LabelType> &lossFunc): dataSet(dataset)
{
    this->lossFunc = &lossFunc;
    //this->eta0 = init_eta;/////////////////////////////////////////
    this->curIterNum = 0;

    this->size_SV=0;
    this->SV_begin=NULL;
    this->SV_end=NULL;
	this->weight=param.weight_sum;
	//this->sigma=sigma_kernel;
}

//////////////////////////////

template <typename FeatType, typename LabelType>
float  Kernel_optim<FeatType, LabelType>::Train()
{
	float errorNum=0;
    if(dataSet.Rewind() == false)
        return 1.f;
    //reset
    while(1)
    {
        const DataChunk<FeatType,LabelType> &chunk = dataSet.GetChunk();
        //all the data has been processed!
        if(chunk.dataNum  == 0)
            break;

        for (size_t i = 0; i < chunk.dataNum; i++)
        {
			if(curIterNum%10000==0)
				cout<<curIterNum<<"\t"<<flush;

            this->curIterNum++;
            const DataPoint<FeatType, LabelType> &data = chunk.data[i];
            float y = this->UpdateWeightVec(data);
            //loss
            if (this->lossFunc->IsCorrect(data.label,y) == false)
            {
                errorNum++;
            }
        }
        dataSet.FinishRead();
    }
	cout<<"\n#Training Instances:"<<curIterNum;
    return errorNum / dataSet.size();
}

//learn a model and return the mistake rate and its variance
template <typename FeatType, typename LabelType>
float  Kernel_optim<FeatType, LabelType>::Learn(float &aveErrRate, float &varErrRate,
        float &sparseRate, int numOfTimes)
{
    float * errorRateVec = new float[numOfTimes];

    for (int i = 0; i < numOfTimes; i++)
    {
        //random order

        errorRateVec[i] = this->Train();
    }
    aveErrRate = Average(errorRateVec, numOfTimes);
    varErrRate = Variance(errorRateVec, numOfTimes);
    sparseRate=1;

    delete []errorRateVec;

    return aveErrRate;
}

//learn a model
template <typename FeatType, typename LabelType>
float  Kernel_optim<FeatType, LabelType>::Learn(int numOfTimes)
{
    float aveErrRate, varErrRate, sparseRate;
    return this->Learn(aveErrRate, varErrRate,sparseRate, numOfTimes);
}//???

//test the performance on the given set
template <typename FeatType, typename LabelType>
float Kernel_optim<FeatType, LabelType>::Test(const Params &param,DataSet<FeatType, LabelType> &testSet)
{
	if(param.ave==0)
	{
		begin_test();
	}
    if(testSet.Rewind() == false)
        exit(0);
    float errorRate(0);
    //test
    while(1)
    {
        const DataChunk<FeatType,LabelType> &chunk = testSet.GetChunk();
        if(chunk.dataNum  == 0) //"all the data has been processed!"
            break;
        for (size_t i = 0; i < chunk.dataNum; i++)
        {
            const DataPoint<FeatType , LabelType> &data = chunk.data[i];
            //predict
            float predict = this->Predict(data);
            if (this->lossFunc->IsCorrect(data.label,predict) == false)
                errorRate++;
        }
        testSet.FinishRead();
    }
    errorRate /= testSet.size();
    return errorRate;
}


template <typename FeatType, typename LabelType>
float Kernel_optim<FeatType, LabelType>::kern(const DataPoint<FeatType, LabelType> &SV_data,const DataPoint<FeatType, LabelType> &x)
{
    float sum=0;
    int i=0;
    int j=0;
    int size_SV_dimension=SV_data.indexes.size();
    int size_data_dimension=x.indexes.size();


    while((i!=size_SV_dimension)&&(j!=size_data_dimension))
    {
        if((SV_data.indexes[i])>(x.indexes[j]))
        {
            sum=sum+x.features[j]*x.features[j];
            j++;
        }
        else if((SV_data.indexes[i])<(x.indexes[j]))
        {
            sum=sum+SV_data.features[i]*SV_data.features[i];
            i++;
        }
        else
        {
            sum=sum+(SV_data.features[i]-x.features[j])*(SV_data.features[i]-x.features[j]);
            i++;
            j++;
        }
    }
    if(i==size_SV_dimension)//i first reach the end
    {
        for(int a=j; a<size_data_dimension; a++)
        {
            sum=sum+x.features[a]*x.features[a];
        }
    }
    if(j==size_data_dimension)//i first reach the end
    {
        for(int a=i; a<size_SV_dimension; a++)
        {
            sum=sum+SV_data.features[a]*SV_data.features[a];
        }
    }


    sum=sum*(-1)*(this->gamma);
    float a=exp(sum);
    return a;

}


template <typename FeatType, typename LabelType>
void Kernel_optim<FeatType, LabelType>::add_SV(SV<FeatType, LabelType> *p_newSV)
{
    if(SV_end!=NULL)
    {
        SV_end->next=p_newSV;
        SV_end=p_newSV;
    }
    else
    {
        SV_begin=p_newSV;
        SV_end=p_newSV;
    }
    size_SV++;
}

template <typename FeatType, typename LabelType>
void Kernel_optim<FeatType, LabelType>::delete_SV(int index_SV)
{
    //index_SV is the index of SV to be deleted from 0 to B-1
    SV<FeatType, LabelType>* p_delete=SV_begin;
    SV<FeatType, LabelType>* q_delete=NULL;
    if((index_SV!=0)&&(index_SV!=size_SV-1))
    {
        int i=0;
        while(i<index_SV-1)
        {
            p_delete=p_delete->next;
            i++;
        }
        q_delete=p_delete->next;
        p_delete->next=q_delete->next;
        delete q_delete;
    }
    else if(index_SV==0)
    {
        SV_begin=p_delete->next;
        delete p_delete;
    }
    else
    {
        int i=0;
        while(i<index_SV-1)
        {
            p_delete=p_delete->next;
            i++;
        }
        q_delete=p_delete->next;
        p_delete->next=NULL;
        delete q_delete;
        SV_end=p_delete;
    }
    size_SV--;
}
template <typename FeatType, typename LabelType>
void  Kernel_optim<FeatType, LabelType>::sum_SV()
{
	//float weight_now =	(float(weight+1))/(float(curIterNum+weight));
	SV<FeatType, LabelType>* p_sum=SV_begin;
	while(p_sum!=NULL)
    {
        //p_sum->SV_alpha_sum=p_sum->SV_alpha_sum*(1-weight_now)+p_sum->SV_alpha*weight_now;
        p_sum->SV_alpha_sum=p_sum->SV_alpha_sum+p_sum->SV_alpha;
		p_sum=p_sum->next;
    }
}



template <typename FeatType, typename LabelType>
  void Kernel_optim<FeatType, LabelType>::Online(Params const & param) {
    if (!dataSet.Rewind()) {
        return;
    }

    float errorRate{0.0f};
    float total{0.0f};
    int block{0};

    double time1 = get_current_time();
    double time2;

    time_predict = 0.0;

  //reset
    while (true) {
        DataChunk<FeatType,LabelType> const & chunk = dataSet.GetChunk();

      //all the data has been processed!
        if (chunk.dataNum  == 0) {
            break;
        }

      // testing stage
        if (this->curIterNum > param.ik_ol_init_block) {
            if(param.ave==0)
            {
                begin_test();
            }

            for (size_t i = 0; i < chunk.dataNum; i++, total++) {
                DataPoint<FeatType, LabelType> const & data = chunk.data[i];
                float predict = this->Predict(data);

                if (this->lossFunc->IsCorrect(data.label,predict) == false) {
                    errorRate++;
                }
            }

            if ((block % param.ik_ol_output_accuracy) == 0) {
                time2 = get_current_time();
                float acc = 1.0f - (errorRate / total);
                std:cout << std::endl << "Accuracy: "  << acc << std::endl;
                std::cout << "Time : " << (time2 - time1) << std::endl;
                std::cout << "Block: " << block << std::endl;
                std::cout << "# Pts: " << this->curIterNum << std::endl;
            }

            block++;
        }

      // training stage
        for (size_t i = 0; i < chunk.dataNum; i++) {
            if ((curIterNum % 10000) == 0) {
                cout << curIterNum << "\t" << flush;
            }

            this->curIterNum++;

            DataPoint<FeatType, LabelType> const & data = chunk.data[i];
            this->UpdateWeightVec(data);
        }

        dataSet.FinishRead();
    }

    time2 = get_current_time();
    double total_time = time2 - time1;
    std::cout << std::endl;
    std::cout << "Final Time: " << total_time << std::endl;
    std::cout << "Final Accuracy: " << ((1.0f - errorRate / total) * 100.0f) << std::endl;

    cout << std::endl << "Training Instances:" << curIterNum << std::endl;

    std::cout << "Total Predict time: " << time_predict << std::endl;
    std::cout << "Predict / Total: " << (time_predict / total_time) << std::endl;
    std::cout << "#SV: " << size_SV << std::endl;
    size_SV = 0; // to stop trying to dealloc non-allocated memory
  }

}
