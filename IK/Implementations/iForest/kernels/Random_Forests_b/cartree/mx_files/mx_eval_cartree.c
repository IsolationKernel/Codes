#include <stdio.h>
#include <mex.h>
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double *Data,*tree_output,*cut_var,*cut_val,*nodechilds,*nodelabel, *nodeheight;
    int i,current_node,M,cvar;
    double mxheight;

    Data= mxGetPr(prhs[0]);
    cut_var = mxGetPr(prhs[1]);
    cut_val = mxGetPr(prhs[2]);
    nodechilds = mxGetPr(prhs[3]);
    nodelabel = mxGetPr(prhs[4]);
    nodeheight = mxGetPr(prhs[5]);
    mxheight = mxGetScalar(prhs[6]);

    M=mxGetM(prhs[0]);
        
    plhs[0] = mxCreateDoubleMatrix(M, 1, mxREAL);
    tree_output = mxGetPr(plhs[0]);
        
    for(i = 0;i<M;i++){
        current_node = 0;
        while (nodechilds[current_node]!=0 && nodeheight[current_node]<= mxheight){
            cvar = cut_var[current_node];
            if (Data[i + (cvar-1)*M] < cut_val[current_node]) current_node = nodechilds[current_node]-1;
            else current_node = nodechilds[current_node];
        }
        tree_output[i] = current_node;
      //  tree_output[i] = nodelabel[current_node];
    }
}
