function dataout = lowmemsub(datamat, ind1, ind2, batchnum)
% This function computes the semi-scatter matrix with low memory usage.

if nargin < 4
    batchnum = 10;
    %the larger the batchnum is, the slower this function works.
end

data_size = length(ind1);
dataout = 0;
for batch_index = 1: batchnum
    batch_size = floor(data_size / batchnum);
    index_begin = (batch_index-1) * batch_size + 1;
    index_end = min(batch_index*batch_size, data_size);
    if batch_index == batchnum
        index_end = data_size;
    end
    datapart = datamat(:,ind1(index_begin:index_end)) - ...
        datamat(:,ind2(index_begin:index_end));
    datapart(isnan(datapart)) = 0;
    dataout = dataout + datapart * datapart.';
end
end