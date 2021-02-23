/*************************************************************************
  > File Name: init_param.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/9/28 15:12:27
  > Functions: init parameters
 ************************************************************************/

#ifndef HEADER_INIT_PARAM
#define HEADER_INIT_PARAM

#include <stdint.h>
#include <string>
namespace SOL {
#define IndexType uint32_t

    //compress cache
#define BASIC_IO 0
#define GZIP_IO 1
#define ZLIB_IO 2
    //
    /////////////////////Optimizer Initalization parameters//////////////////
    //
    //learning rate for sgd
	static const float init_eta = 0.5;
    //budget size
    static const int Budget_ini=100;
    static const float lambda_ini=0.01f;///
    static const float C_bpas_ini=1;
    static const int D_fogd=4*Budget_ini;
    static const float gamma_int=0.01;
    static const float ini_eta_fogd=5e-4f;
    static const int k_nogd_ini= (int)(0.2*Budget_ini);
//	static const int x_ini_dimension=24;///////////////for fogd

    /////////////////////Optimizer Initalization parameters//////////////////
    //
    //whether to learn the best parameter
    static const bool init_is_learn_best_param = false;
    //learning rate

    static const float init_eta_max = 128.f;
    static const float init_eta_min = 1.f;
    static const float init_eta_step = 2.f;
    //pow decaying learing rate
    static const float init_power_t = 0.5;
    //initial t
    static const int init_initial_t = 1;
    //l1 regularization
    static const float init_lambda = 0.001;
    //sparse soft threshold when counting zero-weights
    static const float init_sparse_soft_thresh = (float)(1e-5);
    //truncate gradients every K steps
    static const int init_k = 10;
    //gammarou in enchanced RDA
    static const float init_gammarou = 25;
    //delta in adaptive algorithms
    static const float init_delta = 10;
    static const float init_delta_max = 16.f;
    static const float init_delta_min = 0.125f;
    static const float init_delta_step = 2.f;
    //r in AROW
    static const float init_r = 1;
    static const float init_r_max = 16.f;
    static const float init_r_min = 0.125f;
    static const float init_r_step = 2.f;

    //skip value in SVM2SGD
    static const int init_skip = 16;
    //intial value of norminv in Confidence weighted algorithms
    static const float init_phi =  1.f;
    //is normalize the data
    static const bool init_normalize = false;

    static const char* init_loss_type = "Hinge";
    static const char* init_data_type = "LibSVM";
    static const char* init_opti_method = "SGD";

    //trying the optimal parameters

  // IK parameters
    static const std::string init_ik_model = "DotProduct";
    static const int init_ik_sets = 100;
    static const int init_ik_psi  = 256;

    static const bool init_ik_mode_online = false;

    ////////////////////Data Set Reader Parameters///////////////////////////
    static const size_t init_chunk_size = 256;
    static const size_t init_buf_size = 2;

    //////////////////////Zlib Parameters/////////////////////////////
    static const int zlib_deflate_level = -1; // use default deflate level
    static const size_t zlib_buf_size = 16348; //default buffer size of zlib
}
#endif
