template <typename TLabel>
        void quicksort(double *data, TLabel *labels, int left, int right) {
    
    double pivot, tsd, fl=1;
    int pivot_indx, ll=left, rr=right;
    
    if (left < right) {
        pivot = data[left];
        while (fl==1) {
            while (data[ll] < pivot) ll++;
            while (data[rr] > pivot) rr--;
            if (ll < rr) {
                tsd = data[ll];
                data[ll] = data[rr];
                data[rr] = tsd;
                
                tsd = labels[ll];
                labels[ll] = labels[rr];
                labels[rr] = tsd;
                
                rr--;
                
            }
            else {
                pivot_indx = rr;
                fl=0;
            }
        }
        
        quicksort(data, labels, left, pivot_indx);
        quicksort(data, labels, pivot_indx+1, right);
    }
}


template <typename TLabel>
        void quicksort(double *data, TLabel *labels, double *W, int left, int right) {
    
    
    double pivot, tsd, fl=1;
    int pivot_indx, ll=left, rr=right;
    
    if (left < right) {
        pivot = data[left];
        while (fl==1) {
            while (data[ll] < pivot) ll++;
            while (data[rr] > pivot) rr--;
            if (ll < rr) {
                tsd = data[ll];
                data[ll] = data[rr];
                data[rr] = tsd;
                
                tsd = labels[ll];
                labels[ll] = labels[rr];
                labels[rr] = tsd;
                
                tsd = W[ll];
                W[ll] = W[rr];
                W[rr] = tsd;
                
                rr--;
            }
            else {
                pivot_indx = rr;
                fl=0;
            }
        }
        
        quicksort(data, labels, W, left, pivot_indx);
        quicksort(data, labels, W, pivot_indx+1, right);
    }
}