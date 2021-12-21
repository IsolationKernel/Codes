#include "quickie.h"
#include <cmath>
void GBCP(int M, int N, double* Labels, double* Data, int minleaf, int num_labels, double* bcvar, double* bcval){
    
    double *sorted_data;
    double bh, ch, gr, gl;
    int i, j, cl, nl, mj;
    int *diff_labels_l, *diff_labels_r, *diff_labels, *sorted_labels;
    
    diff_labels_l = new int[num_labels];
    diff_labels_r = new int[num_labels];
    diff_labels = new int[num_labels];
    sorted_labels = new int[M];
    sorted_data =  new double[M];
    
    for(nl=0;nl<num_labels;nl++){
        diff_labels[nl]=0;
    }
    
    for(j = 0;j<M;j++) {
        cl = Labels[j];
        diff_labels[cl-1]++;
    }
    
    bh=0;
    for(nl=0;nl<num_labels;nl++){
        bh+=diff_labels[nl]*diff_labels[nl];
    }
    bh = 1 - (bh/(M*M));
    
    
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
            cl=sorted_labels[j];
            diff_labels_l[--cl]++;
            diff_labels_r[cl]--;
        }
        
        for(j = minleaf-1;j<M-minleaf;j++){
            
            cl=sorted_labels[j];
            diff_labels_l[--cl]++;
            diff_labels_r[cl]--;
            gr = 0;
            gl = 0;
            
            for(nl=0;nl<num_labels;nl++) {
                gl+=diff_labels_l[nl]*diff_labels_l[nl];
                gr+=diff_labels_r[nl]*diff_labels_r[nl];
            }
            gl = 1 - gl/((j+1)*(j+1));
            gr = 1 - gr/((M-j-1)*(M-j-1));
            
            ch = ((j+1)*gl/M) + ((M-j-1)*gr/M);
            
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
}


void GBCP(int M, int N, double* Labels, double* Data, double* W, int minleaf, int num_labels, double* bcvar, double* bcval){
    
    double *sorted_data, *sorted_w;
    double bh, ch, sum_W, sum_l, gr, gl;
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
    
    bh=0;
    for(nl=0;nl<num_labels;nl++){
        bh+=diff_labels[nl]*diff_labels[nl];
    }
    bh = 1 - (bh/(sum_W*sum_W));
    
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
            
            gr = 0;
            gl = 0;
            
            for(nl=0;nl<num_labels;nl++) {
                gl+=diff_labels_l[nl]*diff_labels_l[nl];
                gr+=diff_labels_r[nl]*diff_labels_r[nl];
            }
            gl = 1 - gl/(sum_l*sum_l);
            gr = 1 - gr/((sum_W-sum_l)*(sum_W-sum_l));
            
            ch = ((sum_l)*gl/sum_W) + ((sum_W-sum_l)*gr/sum_W);
            
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
