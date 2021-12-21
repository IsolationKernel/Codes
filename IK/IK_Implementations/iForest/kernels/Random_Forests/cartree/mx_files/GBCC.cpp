#include <cmath>
#include "quickie.h"

void GBCC(int M, int N, double* Labels, double* Data, int minleaf, int num_labels, double* bcvar, double* bcval){
    
    double *saved_logs, *sorted_data;
    double bh, ch;
    int i, j, cl, nl, mj;
    int *diff_labels_l, *diff_labels_r, *diff_labels, *sorted_labels;
    
    diff_labels_l = new int[num_labels];
    diff_labels_r = new int[num_labels];
    diff_labels   = new int[num_labels];
    
    sorted_labels = new int[M];
    saved_logs  =  new double[M];
    sorted_data =  new double[M];
    
    for(nl=0;nl<num_labels;nl++){
        diff_labels[nl]=0;
    }
    
    for(j = 0;j<M;j++) {
        saved_logs[j] = log2(j+1);
        cl = Labels[j];
        diff_labels[cl-1]++;
    }
    
    bh = 0;
    for(nl=0;nl<num_labels;nl++){
        if (diff_labels[nl]>0) bh-=diff_labels[nl]*(saved_logs[diff_labels[nl]-1]-saved_logs[M-1]);
    }
    
    for(i = 0;i<N;i++){
        
        for(nl=0;nl<num_labels;nl++){
            diff_labels_l[nl] = 0;
            diff_labels_r[nl] = diff_labels[nl];
        }
        
        for(j = 0;j<M;j++){
            sorted_data[j] = Data[i*M+j];
            sorted_labels[j] = Labels[j];
        }
        
        quicksort(sorted_data, sorted_labels, 0, M-1);
        
        for(mj = 0 ; mj<minleaf-1;mj++){
            cl=sorted_labels[mj];
            diff_labels_l[--cl]++;
            diff_labels_r[cl]--;
        }
        
        for(j = minleaf-1;j<M-minleaf;j++){
            
            cl=sorted_labels[j];
            diff_labels_l[--cl]++;
            diff_labels_r[cl]--;
            ch = 0;
            
            for(nl=0;nl<num_labels;nl++) {
                if(diff_labels_l[nl]>0) ch-=(diff_labels_l[nl])*(saved_logs[diff_labels_l[nl]-1]-saved_logs[j]);
                if(diff_labels_r[nl]>0) ch-=(diff_labels_r[nl])*(saved_logs[diff_labels_r[nl]-1]-saved_logs[M-j-2]);
            }
            
            if (ch<bh){
                if (fabs(sorted_data[j+1]-sorted_data[j])>1e-15){
                    
                    bh=ch;
                    bcvar[0] = i+1;
                    bcval[0] = 0.5*(sorted_data[j+1]+sorted_data[j]);
                }
            }
        }
    }
    delete[] diff_labels_l;
    delete[] diff_labels_r;
    delete[] diff_labels;
    
    delete[] sorted_labels;
    delete[] sorted_data;
    
    delete[] saved_logs;
}

void GBCC(int M, int N, double* Labels, double* Data, double* W, int minleaf, int num_labels, double* bcvar, double* bcval){
    
    double *sorted_data, *sorted_w;
    double bh, ch, sum_W, sum_l;
    int i, j, cl, nl, mj;
    double *diff_labels_l, *diff_labels_r, *diff_labels;
    int *sorted_labels;
    
    sorted_labels = new int[M];
    sorted_data   = new double[M];
    sorted_w      = new double[M];
    
    diff_labels_l = new double[num_labels];
    diff_labels_r = new double[num_labels];
    diff_labels   = new double[num_labels];
    
    for(nl=0;nl<num_labels;nl++){
        diff_labels[nl]=0;
    }
    
    sum_W=0;
    for(j = 0;j<M;j++) {
        cl = Labels[j];
        diff_labels[cl-1]+=W[j];
        sum_W+=W[j];
    }
    
    bh = 0;
    for(nl=0;nl<num_labels;nl++){
        if (diff_labels[nl]>0) bh-=diff_labels[nl]*(log2(diff_labels[nl])-log2(sum_W));
    }
    
    for(i = 0;i<N;i++){
        
        for(nl=0;nl<num_labels;nl++){
            diff_labels_l[nl] = 0;
            diff_labels_r[nl] = diff_labels[nl];
        }
        
        for(j = 0;j<M;j++){
            sorted_data[j] = Data[i*M+j];
            sorted_labels[j] = Labels[j];
            sorted_w[j] = W[j];
        }
        
        sum_l=0;
        
        quicksort(sorted_data, sorted_labels, sorted_w, 0, M-1);
        
        for(mj = 0 ; mj<minleaf-1;mj++){
            cl=sorted_labels[mj];
            diff_labels_l[--cl]+=sorted_w[mj];
            diff_labels_r[cl]-=sorted_w[mj];
            sum_l+=sorted_w[mj];
        }
        
        for(j = minleaf-1;j<M-minleaf;j++){
            
            cl=sorted_labels[j];
            diff_labels_l[--cl]+=sorted_w[j];
            diff_labels_r[cl]-=sorted_w[j];
            sum_l += sorted_w[j];
            ch = 0;
            
            for(nl=0;nl<num_labels;nl++) {
                if(diff_labels_l[nl]>0) ch-=(diff_labels_l[nl])*(log2(diff_labels_l[nl])-log2(sum_l));
                if(diff_labels_r[nl]>0) ch-=(diff_labels_r[nl])*(log2(diff_labels_r[nl])-log2(sum_W-sum_l));
            }
            
            if (ch<bh){
                if (fabs(sorted_data[j+1]-sorted_data[j])>1e-15){
                    
                    bh=ch;
                    bcvar[0] =i+1;
                    bcval[0] = 0.5*(sorted_data[j+1]+sorted_data[j]);
                }
            }
        }
    }
    delete[] diff_labels_l;
    delete[] diff_labels_r;
    delete[] diff_labels;
    
    delete[] sorted_labels;
    delete[] sorted_data;
    delete[] sorted_w;
}