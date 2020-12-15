#include "mex.h"
#include "iForest.cpp"
class iForest;
void free_model(model_iForest *model){
    for (int i = 0; i < model->num_trees; ++i) {
		std::queue<tree_node *> q_task;
		q_task.push(model->forest[i]);
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
    delete [] model->forest;
    delete [] model->feature_indx;
}
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] ){
    //{"forest", "feature_indx", "psi", "max_height", "num_featurs", "num_trees"};
    int n1 = mxGetM(prhs[0]);
    int f1 = mxGetN(prhs[0]);
    double *dtmp = mxGetPr(prhs[0]);
    double **data1 = new double *[n1];
////#pragma omp parallel for num_threads(8) nowait
    for (int i = 0; i < n1; ++i){
        data1[i] = new double [f1];
        for (int j = 0; j < f1; ++j){
            data1[i][j] = dtmp[j*n1+i];
        }
    }
    
    int n2 = mxGetM(prhs[1]);
    int f2 = mxGetN(prhs[1]);
    dtmp = mxGetPr(prhs[1]);
    double **data2 = new double *[n2];
////#pragma omp parallel for num_threads(8) nowait
    for (int i = 0; i < n2; ++i){
        data2[i] = new double [f2];
        for (int j = 0; j < f2; ++j){
            data2[i][j] = dtmp[j*n2+i];
        }
    }
    
    const mxArray * forest = prhs[2];
    model_iForest * p_model = new model_iForest;
    p_model->psi = (int) mxGetScalar(mxGetFieldByNumber(forest, 0, 2));
    p_model->max_height = (int) mxGetScalar(mxGetFieldByNumber(forest, 0, 3));
    p_model->num_features = (int) mxGetScalar(mxGetFieldByNumber(forest, 0, 4));
    p_model->num_trees = (int) mxGetScalar(mxGetFieldByNumber(forest, 0, 5));
    p_model->feature_indx = new int *[p_model->num_trees];
    int * tmp = (int *)mxGetPr(mxGetFieldByNumber(forest, 0, 1));
   
////#pragma omp parallel for num_threads(8) nowait
    for (int i = 0; i < p_model->num_trees; ++i){
        p_model->feature_indx[i] = new int [p_model->num_features];
        for (int j = 0; j < p_model->num_features; ++j){
            p_model->feature_indx[i][j] = tmp[j * p_model->num_trees + i];
        }
    }
    mxArray * mx = mxGetFieldByNumber(forest, 0, 0);
    p_model->forest = new tree_node*[p_model->num_trees];
//#pragma omp parallel for num_threads(8) nowait
    for (int i = 0; i < p_model->num_trees; ++i){
        mxArray * mx2 = mxGetFieldByNumber(mx,i, 0);
        p_model->forest[i] = new tree_node;
        std::queue<mxArray *> q_task;
        std::queue<tree_node *> q_task_c;
        q_task.push(mx2);
        q_task_c.push(p_model->forest[i]);
        while (q_task.size()!=0){
            mxArray *p_struct = q_task.front();
            tree_node *p_node = q_task_c.front();
            q_task.pop();
            q_task_c.pop();
            //{"is_leaf", "height", "node_size", "p_indx_data",  "left_child", "right_child", "split_feature_id", "threshold"};
            p_node->is_leaf = (bool) mxGetScalar(mxGetFieldByNumber(p_struct, 0, 0));
            p_node->height = (int) mxGetScalar(mxGetFieldByNumber(p_struct, 0, 1));
            p_node->node_size = (int) mxGetScalar(mxGetFieldByNumber(p_struct, 0, 2));
            p_node->p_indx_data = new int [p_node->node_size];
            memcpy(p_node->p_indx_data, mxGetPr(mxGetFieldByNumber(p_struct, 0, 3)), sizeof(int)*p_node->node_size);
            if (!p_node->is_leaf){
                p_node->left_child = new tree_node;
                p_node->right_child = new tree_node;
                q_task.push(mxGetFieldByNumber(p_struct, 0, 4));
                q_task.push(mxGetFieldByNumber(p_struct, 0, 5));
                q_task_c.push(p_node->left_child);
                q_task_c.push(p_node->right_child);
            }
            p_node->split_feature_id = (int) mxGetScalar(mxGetFieldByNumber(p_struct, 0, 6));
            p_node->threshold = (double) mxGetScalar(mxGetFieldByNumber(p_struct, 0, 7));
        }
    }
    iForest *p_forest = new iForest; 
    p_forest->set_model(p_model);
    double ** sim = p_forest->get_relative_similarity(data1, n1, data2, n2);
    plhs[0] = mxCreateNumericMatrix(n1, n2, mxDOUBLE_CLASS, mxREAL);
    double * pr = mxGetPr(plhs[0]);
////#pragma omp parallel for num_threads(8) nowait
    for (int i = 0; i < n1; ++i){
        for (int j = 0; j < n2; ++j){
            pr[j * n1 + i] = sim[i][j];
        }
    }
    
    for (int i = 0; i < n1; ++i){    
       delete [] sim[i];
    }
    delete sim;
    
    for (int i = 0; i < n1; ++i){
        delete [] data1[i];
    }
    delete data1;
    for (int i = 0; i < n2; ++i){
        delete [] data2[i];
    }
    delete data2;
 //   delete p_model;
 //   free_model(p_model);
    delete p_forest;
   delete p_model;
}