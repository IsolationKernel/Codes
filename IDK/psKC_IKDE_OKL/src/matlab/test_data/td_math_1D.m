clear;

x = csvread('./td_math_1D_10a.csv');
y = csvread('./td_math_1D_10b.csv');

dist = pdist2(x, y);

dlmwrite('./res_math_1D_dist_p_2.csv', dist, 'delimiter', ',', 'precision', 20);

[idx, D] = knnsearch(y,x);
idx = idx - 1;

csvwrite('./res_math_1D_nn_psi_10.csv', idx);

idx_all = [];

for i = 1:10
    idx = [];
    for j = 0:1
        start = 5 * j + 1;
        stop = start + 4;
        res = knnsearch(y(start:stop), x(i)) - 1;
        idx = [idx res];
    end
    idx_all = vertcat(idx_all, idx);
end

csvwrite('./res_math_1D_nn_psi_5.csv', idx_all);
