clear;

x = csvread('./td_math_1000D_10000a.csv');
y = csvread('./td_math_1000D_20000b.csv');

dist = pdist2(x, y);

dlmwrite('./res_math_1000D_dist_p_2.csv', dist, 'delimiter', ',', 'precision', 20);
