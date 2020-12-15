#include <string.h>
#include <ctype.h>
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 
//#include <omp.h> 
#include "iForest.h"
#include <queue>

iForest::iForest() {
	//	this->_init();
}
iForest::iForest(int psi, int max_height, int num_features, int num_trees) {
	this->m_psi = psi;
	this->m_max_height = max_height;
	this->m_num_features = num_features;
	this->m_num_trees = num_trees;
	//	this->_init();
}
iForest::iForest(int psi, int num_trees) {
	this->m_psi = psi;
	this->m_num_trees = num_trees;
	this->m_max_height = (int)round(log(psi) / log(2));
	//	this->_init();
}
iForest::iForest(int psi) {
	this->m_psi = psi;
	this->m_max_height = (int)round(log(psi) / log(2));
	//	this->_init();
}
iForest::~iForest() {
    if (this->m_forest){
        for (int i = 0; i < this->m_num_trees; ++i) {
            std::queue<tree_node *> q_task;
            if (this->m_forest[i]){
                q_task.push(this->m_forest[i]);
            }
            while (q_task.size() != 0) {
                tree_node * ptr_node = q_task.front();
                q_task.pop();
                if (ptr_node->left_child != NULL) {
                    q_task.push(ptr_node->left_child);
                }
                if (ptr_node->right_child != NULL) {
                    q_task.push(ptr_node->right_child);
                }
                if (ptr_node->p_indx_data != NULL) {
                    delete[] ptr_node->p_indx_data;
                }
                delete ptr_node;
            }
        }
    }
    if (this->m_forest){
        delete [] this->m_forest;
    }
    if (this->m_feature_indx){
        delete [] this->m_feature_indx;
    }
}
void iForest::_init(int n, int d) {
	this->m_psi = this->m_psi < n ? this->m_psi : n;
	if (this->m_num_features == 0) {
		this->m_num_features = d;
	}
	else {
		this->m_num_features = this->m_num_features < d ? this->m_num_features : d;
	}
	this->m_forest = new tree_node *[this->m_num_trees];
	this->m_feature_indx = new int *[this->m_num_trees];

	for (int i = 0; i < this->m_num_trees; ++i) {
		this->m_forest[i] = new tree_node;
		this->m_feature_indx[i] = new int[this->m_num_features];
		int * feature_indx = new int[d];
		for (int j = 0; j < d; ++j) {
			feature_indx[j] = j;
		}
		this->_rand_sample(feature_indx, d, this->m_feature_indx[i], this->m_num_features);
		delete[] feature_indx;
	}
}
double iForest::_max(double **data, int *indx_data, int n, int fid) {
	double tmp = -999999;
	if (indx_data == NULL) {
		return tmp;
	}
	for (int i = 0; i < n; ++i) {
		int indx = indx_data[i];
		if (data[indx][fid] > tmp) {
			tmp = data[indx][fid];
		}
	}
	return tmp;
}
double iForest::_min(double **data, int *indx_data, int n, int fid) {
	double tmp = 999999;
	if (indx_data == NULL) {
		return tmp;
	}
	for (int i = 0; i < n; ++i) {
		int indx = indx_data[i];
		if (data[indx][fid] < tmp) {
			tmp = data[indx][fid];
		}
	}
	return tmp;
}
void iForest::_rand_sample(int *indx1, int n, int *indx2, int m)
{
    int * tmp = new int[n];
    memcpy(tmp, indx1, sizeof(int)*n);
	std::random_shuffle(tmp, tmp + n);
    if (indx2 == NULL) {
		indx2 = new int[m];
	}
	for (int i = 0; i < m; ++i) {
		indx2[i] = tmp[i];
	}
    delete[] tmp;
// 	int select = m;
// 	int remaining = n;
// 	int j = 0;
// 	if (indx2 == NULL) {
// 		indx2 = new int[m];
// 	}
// 	for (int i = 0; i < n; ++i)
// 	{
// 		// ÿһ��ѭ���������i��ѭ�������������ѡ�еĸ���
// 		// ���쵽��i��ʱ���ɼ�֮ǰ�Ѿ����˹�i�ε��жϣ���0��ʼ����
// 		// Ҳ���������µ�(n-i)��ֵ���ҵ�i���ֵĸ���
// 		if (rand() % remaining < select)
// 		{
// 			indx2[j++] = indx1[i];
// 			--select;
// 		}
// 		--remaining;
// 	}
}

void iForest::_build_isolation_tree(tree_node *tree, int * feature_indx, double **data, int n, int d) {
	std::queue<tree_node *> q_task;
	tree->height = 1;
	int * indx_all_data = new int[n];
	for (int i = 0; i < n; ++i) {
		indx_all_data[i] = i;
	}
	tree->p_indx_data = new int[this->m_psi];
	this->_rand_sample(indx_all_data, n, tree->p_indx_data, this->m_psi);
	tree->node_size = this->m_psi;
	q_task.push(tree);
	while (q_task.size() != 0) {
		tree_node * ptr_node = q_task.front();
		q_task.pop();
		if (ptr_node->height >= this->m_max_height) {
			ptr_node->is_leaf = true;
			ptr_node->left_child = NULL;
			ptr_node->right_child = NULL;
			continue;
		}
		if (ptr_node->node_size <= 1) {
			ptr_node->is_leaf = true;
			ptr_node->left_child = NULL;
			ptr_node->right_child = NULL;
			continue;
		}
		double mx_fid, mn_fid;
		for (int j = 0; j < 10; ++j) {
			int * fid = new int[2];
			this->_rand_sample(feature_indx, this->m_num_features, fid, 2);
			ptr_node->split_feature_id = fid[0];
			delete[] fid;
			mx_fid = this->_max(data, ptr_node->p_indx_data, ptr_node->node_size, ptr_node->split_feature_id);
			mn_fid = this->_min(data, ptr_node->p_indx_data, ptr_node->node_size, ptr_node->split_feature_id);
			if (mx_fid != mn_fid) {
				ptr_node->threshold = mn_fid + (mx_fid - mn_fid) * (rand() % 10000 + 1)*1.0 / 10001.0;
				break;
			}
		}
		if (mx_fid == mn_fid) {
			ptr_node->is_leaf = true;
			ptr_node->left_child = NULL;
			ptr_node->right_child = NULL;
			continue;
		}

		ptr_node->left_child = new tree_node;
		ptr_node->left_child->height = ptr_node->height + 1;
		ptr_node->right_child = new tree_node;
		ptr_node->right_child->height = ptr_node->height + 1;

		int lcount = 0, rcount = 0;
		for (int j = 0; j < ptr_node->node_size; ++j) {
			int didx = ptr_node->p_indx_data[j];
			int fidx = ptr_node->split_feature_id;
			if (data[didx][fidx] < ptr_node->threshold) {
				lcount++;
			}
			else {
				rcount++;
			}
		}
		ptr_node->left_child->node_size = lcount;
		ptr_node->right_child->node_size = rcount;
		ptr_node->left_child->p_indx_data = new int[lcount];
		ptr_node->right_child->p_indx_data = new int[rcount];
		lcount = 0; rcount = 0;
		for (int j = 0; j < ptr_node->node_size; ++j) {
			int didx = ptr_node->p_indx_data[j];
			int fidx = ptr_node->split_feature_id;
			if (data[didx][fidx] < ptr_node->threshold) {
				ptr_node->left_child->p_indx_data[lcount++] = ptr_node->p_indx_data[j];
			}
			else {
				ptr_node->right_child->p_indx_data[rcount++] = ptr_node->p_indx_data[j];
			}
		}
		q_task.push(ptr_node->left_child);
		q_task.push(ptr_node->right_child);
	}
	delete[] indx_all_data;
}

	

model_iForest * iForest::build_iForest(double **data, int n, int d) {
	this->_init(n, d);
//    #pragma omp parallel for num_threads(8) nowait
	for (int i = 0; i < this->m_num_trees; ++i) {
		this->_build_isolation_tree(this->m_forest[i], this->m_feature_indx[i] ,data, n, d);
	}
	return this->get_model();
}
model_iForest* iForest::get_model() {
	model_iForest* model = new model_iForest;
	model->forest = this->m_forest;
	model->feature_indx = this->m_feature_indx;
	model->psi = this->m_psi;
	model->max_height = this->m_max_height;
	model->num_features = this->m_num_features;
	model->num_trees = this->m_num_trees;
	return model;
}

double iForest::get_relative_similarity(double * inst1, double * inst2) {
	int common = 0;
	int spec1 = 0;
	int spec2 = 0;
	double sim = 0.0;
	tree_node * p1, *p2;
	// #pragma omp parallel for num_threads(8)
	for (int i = 0; i < this->m_num_trees; ++i) {
		p1 = this->m_forest[i];
		p2 = p1;
		while (p1 == p2 && p1 && p2) {
			common = p1->node_size;
			spec1 = p1->node_size;
			spec2 = p2->node_size;
			if (p1 && p2 && p1->is_leaf && p2->is_leaf && p1 == p2) {
				sim = sim + 1;
				break;
			}
			if (inst1[p1->split_feature_id] <= p1->threshold) {
				p1 = p1->left_child;
			}
			else {
				p1 = p1->right_child;
			}
			if (inst2[p2->split_feature_id] <= p2->threshold) {
				p2 = p2->left_child;
			}
			else {
				p2 = p2->right_child;
			}
		}
	}
	return sim / this->m_num_trees;
}
double ** iForest::get_relative_similarity(double ** data1, int n1, double ** data2, int n2){
    
    double ** sim = new double *[n1];
//   #pragma omp parallel for num_threads(8) nowait
    for (int i = 0; i < n1; ++i){
        sim[i] = new double [n2];
        for (int j = 0; j < n2; ++j){
            sim[i][j] = this->get_relative_similarity(data1[i], data2[j]);
         }
    }
    return sim;
}
void iForest::set_model(model_iForest *p_model){
    this->m_forest = p_model->forest;
    this->m_psi = p_model->psi;
    this->m_max_height = p_model->max_height;
    this->m_num_features = p_model->num_features;
    this->m_num_trees = p_model->num_trees;
    this->m_feature_indx = p_model->feature_indx;
}