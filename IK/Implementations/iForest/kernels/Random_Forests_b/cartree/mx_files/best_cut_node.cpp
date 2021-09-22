#include <mex.h>
#include "node_cuts.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double *Data, *Labels, *W, *bcvar, *bcval;
    int N, M, num_labels, minleaf;
    char *method;
    
    
    method = (char*) malloc(2*sizeof(char));
    mxGetString(prhs[0], method, 2);
    
    Data = mxGetPr(prhs[1]);
    
    N=mxGetN(prhs[1]);
    M=mxGetM(prhs[1]);
    
    Labels = mxGetPr(prhs[2]);
    W = mxGetPr(prhs[3]);
    minleaf = mxGetScalar(prhs[4]);
    
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    bcvar = mxGetPr(plhs[0]);
    bcvar[0]=-1;
    
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    bcval = mxGetPr(plhs[1]);
    
    if ((method[0] == 'c')  || (method[0] == 'g')) num_labels = mxGetScalar(prhs[5]);
    
    if (mxGetN(prhs[3])==0){
        switch (method[0]){
            case 'c':
                GBCC(M, N, Labels, Data, minleaf, num_labels, bcvar, bcval);
                break;
            case 'g':
                GBCP(M, N, Labels, Data, minleaf, num_labels, bcvar, bcval);
                break;
            case 'r':
                GBCR(M, N, Labels, Data, minleaf, bcvar, bcval);
                break;
        }
    }
    else{
        switch (method[0]){
            case 'c':
                GBCC(M, N, Labels, Data, W, minleaf, num_labels, bcvar, bcval);
                break;
            case 'g':
                GBCP(M, N, Labels, Data, W, minleaf, num_labels, bcvar, bcval);
                break;
            case 'r':
                GBCR(M, N, Labels, Data, W, minleaf, bcvar, bcval);
                break;
        }
    }
    
    delete[] method;
}