function A = MetricLearningAutotuneKnn(y, X, k, params)
% This function runs geometric mean metric learning over various parameters
% of t, choosing that with the highest accuracy.

% You can replace the following approach of finding the best t with any
% heuristic method you have in your mind.

if (~exist('params')),
    params = struct();
end
params = SetDefaultParams(params);
f = params.tuning_num_fold;

t = 0.1:0.2:0.9;
for i = 1:length(t)
    ac(i) = CrossValidateKNN(y, X, @(y,X) MetricLearning(y, X, t(i), params), f, k);
end
[v,o] = max(ac);

if (t(o) == 0.1) || (t(o) == 0)
    t2 = [linspace(0,0.24,13),0.00001,0.001];
elseif (t(o) == 0.9) || (t(o) == 1)
    t2 = [linspace(0.76,1,13),0.9999,0.995];
else
    t2 = (t(o)-0.14):0.02:(t(o)+0.14);
end

for i = 1:length(t2)
    ac2(i) = CrossValidateKNN(y, X, @(y,X) MetricLearning(y, X, t2(i), params), f, k);
end
[v2,o2] = max(ac2);
t_opt = t2(o2);

disp(sprintf('\tThe optimal value of t: %f', t_opt));

A = MetricLearning(y, X, t_opt, params);
end