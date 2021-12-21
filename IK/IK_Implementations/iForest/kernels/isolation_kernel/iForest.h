#pragma once
#include <stdio.h>
struct tree_node;
struct model_iForest;
class iForest
{  
public:
	int m_psi = 256;
	int m_max_height = 8;
	int m_num_features = 0;
	int m_num_trees = 100;
	tree_node **m_forest;
	int ** m_feature_indx;
	void _init(int n, int d);
	void _rand_sample(int *indx1, int n, int *indx2, int m);
	double _max(double **data, int * indx_data, int n, int fid);
	double _min(double **data, int *indx_data, int n, int fid);
 	void _build_isolation_tree(tree_node *tree, int * feature_indx, double **data, int n, int d);
public:
	iForest();
	iForest(int psi, int max_height, int num_features, int num_trees);
	iForest(int psi, int num_trees);
	iForest(int psi);
	~iForest();
    double get_split_value(double ** data, int * indx_data, int n, int fid); 
	model_iForest * build_iForest(double **data, int n, int d);
	model_iForest * get_model();
    double get_relative_similarity(double * inst1, double * inst2);
    double ** get_relative_similarity(double ** data1, int n1, double ** data2, int n2);
    void set_model(model_iForest *p_model);
};

struct tree_node {
	bool is_leaf = false;
	int height = 0;
	int node_size = 0;
	int *p_indx_data = NULL;
	tree_node * left_child = NULL;
	tree_node * right_child = NULL;
	int split_feature_id = -1;
	double threshold = 0;
};

struct model_iForest {
	tree_node ** forest = NULL;
	int ** feature_indx = NULL;
	int psi = 256;
	int max_height = 8;
	int num_features = 0;
	int num_trees = 0;
};
