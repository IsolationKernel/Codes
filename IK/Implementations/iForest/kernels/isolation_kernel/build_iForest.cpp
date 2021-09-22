#include "mex.h"
#include "iForest.cpp"
void free_model(model_iForest *model);
mxArray *node_mxArray(tree_node *node){
    const int num_fields = 8;
    const char *field_names[8] = {"is_leaf", "height", "node_size", "p_indx_data",  "left_child", "right_child", "split_feature_id", "threshold"};
    const mwIndex index = 0;
    mxArray *mx;
    mxArray *mx_sub;
    mx = mxCreateStructMatrix(1, 1, num_fields, field_names);
    mx_sub = mxCreateLogicalScalar(node->is_leaf);
    memcpy(mxGetPr(mx_sub), &(node->is_leaf), sizeof(bool));
    mxSetFieldByNumber(mx, 0, 0, mx_sub);
    
    mx_sub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx_sub), &(node->height), sizeof(int));
    mxSetFieldByNumber(mx, 0, 1, mx_sub);
    
    mx_sub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx_sub), &(node->node_size), sizeof(int));
    mxSetFieldByNumber(mx, 0, 2, mx_sub);
    
    mx_sub = mxCreateNumericMatrix(1, node->node_size, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx_sub), node->p_indx_data, sizeof(int)*node->node_size);
    mxSetFieldByNumber(mx, 0,  3, mx_sub);
    
    mx_sub = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx_sub), &(node->split_feature_id), sizeof(int));
    mxSetFieldByNumber(mx, 0,  6, mx_sub);
    
    mx_sub = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    memcpy(mxGetPr(mx_sub), &(node->threshold), sizeof(double));
    mxSetFieldByNumber(mx, 0,  7, mx_sub);
    
    if (node->is_leaf){
        mx_sub = mxCreateStructMatrix(0, 0, num_fields, field_names);
        mxSetFieldByNumber(mx, 0,  4, mx_sub);
        
        mx_sub = mxCreateStructMatrix(0, 0, num_fields, field_names);
        mxSetFieldByNumber(mx, 0,  5, mx_sub);
    } else{
        mx_sub = node_mxArray(node->left_child);
        mxSetFieldByNumber(mx, 0,  4, mx_sub);
        mx_sub = node_mxArray(node->right_child);
        mxSetFieldByNumber(mx, 0,  5, mx_sub);
    }
    return mx;
}

mxArray *model_mxArray(model_iForest * model){
    mxArray *pm, *mx, *mx_sub;
    const int num_fields = 6;
    const char *field_names[6] = {"forest", "feature_indx", "psi", "max_height", "num_featurs", "num_trees"};
    pm = mxCreateStructMatrix(1, 1, num_fields, field_names);
    
    const int num_fields_tree = 1;
    const char *field_names_tree[1] = {"tree"};
    mx = mxCreateStructMatrix(model->num_trees, 1, num_fields_tree, field_names_tree);
    for (int i = 0; i < model->num_trees; ++i){
        mx_sub = node_mxArray(model->forest[i]);
        mxSetFieldByNumber(mx, i, 0, mx_sub);
    }
    mxSetFieldByNumber(pm, 0, 0, mx);
    
    mx = mxCreateNumericMatrix(model->num_trees, model->num_features, mxINT32_CLASS, mxREAL);
    //memcpy(mxGetPr(mx), &(model->feature_indx), sizeof(int)*model->num_features*model->num_trees);
    int * tmp = (int *)mxGetPr(mx);
    //#pragma omp parallel for num_threads(8) nowait
    for (int i = 0 ; i < model->num_trees; ++i){
           for (int j = 0; j < model->num_features; ++j){
                tmp[j*model->num_trees+i] = model->feature_indx[i][j];
           }
    } 
    mxSetFieldByNumber(pm, 0, 1, mx);
    
    mx = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx), &(model->psi), sizeof(int));
    mxSetFieldByNumber(pm, 0, 2, mx);
    
    
    mx = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx), &(model->max_height), sizeof(int));
    mxSetFieldByNumber(pm, 0, 3, mx);
    
    mx = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx), &(model->num_features), sizeof(int));
    mxSetFieldByNumber(pm, 0, 4, mx);
    
    mx = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(mx), &(model->num_trees), sizeof(int));
    mxSetFieldByNumber(pm, 0, 5, mx);   
    
  
    return pm;
}

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
			if (ptr_node) {
                delete ptr_node;
            }
		}
	}
    delete [] model->forest;
    delete [] model->feature_indx;
}


void mexFunction( int nlhs, mxArray *plhs[],
	  int nrhs, const mxArray *prhs[] ){
     
      

      
      iForest *p_forest; 
      int psi, max_height, num_trees, num_features;
      switch(nrhs){
          case 0:
              printf("! Data matrix is required\n");
              return;
          case 1: 
              p_forest = new iForest();
              break;
          case 2:
              psi = (int)mxGetScalar(prhs[1]);
              p_forest = new iForest(psi);
              break;
          case 3:
              psi = (int)mxGetScalar(prhs[1]);
              num_trees = (int)mxGetScalar(prhs[2]);
              p_forest = new iForest(psi, num_trees);
              break;
          case 4:
              printf("! Input does not match\n");
              return;
          case 5:
              psi = (int)mxGetScalar(prhs[1]);
              num_trees = (int)mxGetScalar(prhs[2]);
              max_height = (int)mxGetScalar(prhs[3]);
              num_features = (int)mxGetScalar(prhs[4]);   
           //   printf("%d %d %d %d", psi, num_trees, max_height, num_features);
              p_forest = new iForest(psi,max_height,num_features, num_trees);
              break;
      }
      double * mx = mxGetPr(prhs[0]);
      int r = mxGetM(prhs[0]);
      int c = mxGetN(prhs[0]);
      double ** data = new double*[r];
     //#pragma omp parallel for num_threads(8) nowait
      for (int i = 0; i < r; ++i){
          data[i] = new double [c];             
          for (int j = 0; j < c; ++j){
              data[i][j] = mx[j * r + i];
          }
      }  

      model_iForest* model = p_forest->build_iForest(data, r, c);
      
//      double ** sim = p_forest->get_relative_similarity(data, r, data, r);
//       for (int i = 0 ; i < model->num_trees; ++i){
//            for (int j = 0; j < model->num_features; ++j){
//                 printf("%d ", model->feature_indx[i][j]);
//            }
//            printf("\n");
//       }
      
//      forest.print_model(model);
      
      if (model != NULL){
          plhs[0] = model_mxArray(model);
       //   free_model(model);
      }  
      
//       plhs[1] = mxCreateNumericMatrix(r, r, mxDOUBLE_CLASS, mxREAL);
//       double * pr = mxGetPr(plhs[1]);
//       #pragma omp parallel for num_threads(8)
//       for (int i = 0; i < r; ++i){
//           for (int j = 0; j < r; ++j){
//               pr[j * r + i] = sim[i][j];
//           }
//       }
      if (p_forest != NULL){
          delete p_forest;
      }

      if (model != NULL){
         // free_model(model);
          if (model) {
              delete model;
          }
      }
      for (int i = 0; i < r; ++i){           
          delete [] data[i];
      }
      delete data;   
}