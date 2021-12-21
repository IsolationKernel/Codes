/*************************************************************************
> File Name: Params.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 26 Sep 2013 05:51:05 PM SGT
> Functions: Class for Parsing parameters
************************************************************************/

#ifndef HEADER_PARSER_PARAM
#define HEADER_PARSER_PARAM

#include "common/ezOptionParser.hpp"

#include "data/parser.h"

#include <string>
#include <map>


using std::string;
using std::map;

//using namespace ez;

namespace SOL
{
	class Params
	{
	private:
		ez::ezOptionParser opt;
		ez::ezOptionValidator* vfloat; 
		ez::ezOptionValidator* vint; 
		ez::ezOptionValidator* vbool; 

		map<std::string, float*> flag2storage_float;
		map<std::string, int*> flag2storage_int;
		map<std::string, bool*> flag2storage_bool;
		map<std::string, std::string*> flag2storage_str;

		typedef map<std::string, float*>::iterator map_float_iter;
		typedef map<std::string, int*>::iterator map_int_iter;
		typedef map<std::string, bool*>::iterator map_bool_iter;
		typedef map<std::string, std::string*>::iterator map_str_iter;

	public:
		//input data
		string fileName; //source file name
		string cache_fileName; //cached file name
		string test_fileName; //test file name
		string test_cache_fileName; //cached test file name

		//dataset type
		string str_data_type;
		//loss function type
		string str_loss;
		//optimization method
		string str_opt;

		int passNum;
		int D_set;
		bool ave;

		//optimzation parameters
		float eta; //learning rate
		float eta1;
		float gamma;
		float lambda; //for l1 regularization
		int K; //for STG method
		int Budget_set;
		float gamma_rou; //for RDA
		int k_nogd;
		float delta; //for Ada-
		float r; //for AROW
		float phi; //for SCW
		float C;
		int buf_size; //number of chunks in dataset 
		int start_ave;
		float C_bpas;

		int initial_t;
		float power_t; 
		bool is_learn_best_param; //whether learn best parameter

		bool is_normalize;

		float beta_spa;
		float alpha_spa;
		int weight_sum;
		float delt_max;

      // IK parameters
        string ik_model;
        int ik_sets;
        int ik_psi;

        bool ik_mode_online;
        int  ik_ol_init_block;
        int  ik_ol_output_accuracy;

    public:
		Params();
		~Params();

		bool Parse(int argc, const char** args);
		void Help();

	private:
		void Init();

		void add_option(float default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, float *storage);
		void add_option(int default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, int *storage);
		void add_option(bool default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, bool *storage);
		void add_option(const char* default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, string *storage);
	};
}
#endif
