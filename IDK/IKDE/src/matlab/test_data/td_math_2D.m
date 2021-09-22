clear;

x = csvread('./td_math_2D_100a.csv');
y = csvread('./td_math_2D_100b.csv');

dist = pdist2(x, y);

dlmwrite('./res_math_2D_dist_p_2.csv', dist, 'delimiter', ',', 'precision', 20);
