/*************************************************************************
    > File Name: main.cpp
    > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
    > Created Time: 2013/9/20 13:18:02
    > Functions:
 ************************************************************************/
#include "Params.h"
#include "common/util.h"

#include "data/DataSet.h"
#include "data/libsvmread.h"

#include "loss/LogisticLoss.h"
#include "loss/HingeLoss.h"
#include "loss/SquareLoss.h"
#include "loss/SquaredHingeLoss.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

#include "kernel/kernel_optim.h"
#include "kernel/kernel_perceptron.h"
#include "kernel/kernel_sgd.h"
#include "kernel/kernel_RBP.h"
#include "kernel/kernel_forgetron.h"
#include "kernel/kernel_projectron.h"
#include "kernel/kernel_projectronpp.h"
#include "kernel/kernel_bogd.h"
#include "kernel/kernel_bpas.h"
#include "kernel/kernel_nogd.h"
#include "kernel/kernel_fogd.h"
#include "kernel/kernel_pa.h"

#include "kernel/kernel_ik_ogd.h"

using namespace std;
using namespace SOL;

#define FeatType float
#define LabelType char

///////////////////////////function declarications/////////////////////
void FakeInput(int &argc, char **args, char** &argv);
template <typename T1, typename T2> LossFunction<T1,T2>* GetLossFunc(const Params &param);
template <typename T1, typename T2>
Kernel_optim<T1,T2>* GetOptimizer(const Params &param, DataSet<T1,T2> &dataset, LossFunction<T1,T2> &lossFun);
///////////////////
int main(int argc, const char** args) {

   //check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
    int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
    tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
    _CrtSetDbgFlag( tmpFlag );
#endif
    Params param;
    if (param.Parse(argc, args) == false){
        return -1;
    }

    LossFunction<FeatType, LabelType> *lossFunc = GetLossFunc<FeatType, LabelType>(param);
    if(lossFunc == NULL)
        return -1;

    DataSet<FeatType, LabelType> dataset(param.passNum,param.buf_size);
    if (dataset.Load(param.fileName, param.cache_fileName) == false){
        cerr<<"ERROR: Load dataset "<<param.fileName<<" failed!"<<endl;
        delete lossFunc;
        return -1;
    }

    Kernel_optim<FeatType, LabelType> *opti = GetOptimizer(param,dataset,*lossFunc);
    if (opti == NULL)
        return -1;

    opti->SetParameter(param.gamma,param.eta);

    opti->PrintOptInfo();

    if (param.ik_mode_online) {
        if (param.ik_ol_init_block == -1) {
            std::cerr << "No initial block size given for online mode. Terminating ..." << std::endl;
        } else {
            opti->Online(param);
        }
    } else {
        float l_errRate(0), l_varErr(0);	//learning error rate
        float sparseRate(0);
        //learning the model
        double time1 = get_current_time();

        opti->Learn(l_errRate,l_varErr,sparseRate);

        double time2 = get_current_time();

        printf("\nLearn acuracy: %.6f%%\n",(1-l_errRate)* 100);
        cout<<"#SV:"<<opti->size_SV<<endl;
        double time3 = 0;
        printf("Learning time: %.6f s\n", (float)(time2 - time1));


        //test the model
        bool is_test = param.test_cache_fileName.length() > 0 || param.test_fileName.length() > 0;
        if ( is_test) {
            DataSet<FeatType, LabelType> testset(1,param.buf_size);
            if (testset.Load(param.test_fileName, param.test_cache_fileName) == true) {
                float t_errRate(0);	//test error rate
                t_errRate = opti->Test(param,testset);
                time3 = get_current_time();

                printf("Test acuracy: %.6f %%\n",(1-t_errRate) * 100);
            }
            else
                cout<<"load test set failed!"<<endl;
        }


        if (is_test)
            printf("Test time: %.6f s\n", (float)(time3 - time2));

    }

    delete lossFunc;
    delete opti;

    return 0;
}

template <typename T1, typename T2>
LossFunction<T1,T2>* GetLossFunc(const Params &param) {
    if (param.str_loss == "Hinge")
        return new HingeLoss<T1,T2>();
    else if (param.str_loss == "Logit")
        return new LogisticLoss<T1,T2>();
    else if (param.str_loss == "Square")
        return new SquareLoss<T1,T2>();
    else if (param.str_loss == "SquareHinge")
        return new SquaredHingeLoss<T1, T2>();
    else{
        cerr<<"ERROR: unrecognized Loss function "<<param.str_loss<<endl;
        return NULL;
    }
}


template <typename T1, typename T2>
Kernel_optim<T1,T2>* GetOptimizer(const Params &param, DataSet<T1,T2> &dataset, LossFunction<T1,T2> &lossFunc) {
    string method = param.str_opt;
    ToUpperCase(method);
    const char* c_str = method.c_str();
    if (strcmp(c_str, "KERNEL-PERCEPTRON") == 0)
        return new kernel_perceptron<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-OGD") == 0)
        return new kernel_sgd<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-RBP") == 0)
        return new kernel_RBP<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-FORGETRON") == 0)
        return new kernel_forgetron<T1, T2>(param,dataset,lossFunc);

    else if (strcmp(c_str, "KERNEL-PROJECTRON") == 0)
        return new kernel_projectron<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-PROJECTRONPP") == 0)
        return new kernel_projectronpp<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-BOGD") == 0)
        return new kernel_bogd<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-BPAS") == 0)
        return new kernel_bpas<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-FOGD") == 0)
        return new kernel_fogd<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-NOGD") == 0)
        return new kernel_nogd<T1, T2>(param,dataset,lossFunc);
    else if (strcmp(c_str, "KERNEL-IK_OGD") == 0)
        return new kernel_ik_ogd<T1, T2>(param,dataset,lossFunc);
    else{
        cerr<<"ERROR: unrecgonized optimization method "<<param.str_opt<<endl;
        return NULL;
    }
}
