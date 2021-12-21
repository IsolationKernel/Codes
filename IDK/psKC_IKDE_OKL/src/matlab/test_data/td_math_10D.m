clear;

x = csvread('./td_math_10D_1000a.csv');
y = csvread('./td_math_10D_1000b.csv');

dist = pdist2(x, y);

dlmwrite('./res_math_10D_dist_p_2.csv', dist, 'delimiter', ',', 'precision', 20);
