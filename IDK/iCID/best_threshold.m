function[best_threshold, result] = best_threshold(Pscore, a)
% a is alpha, Pscore is point dissimilarity score
% Adjust the threshold by changing the value of a

best_threshold = mean(Pscore)+a*std(Pscore);

% the point will be labelled as 1 if Pscore larger than threshold.
% the point will be labelled as 0 if Pscore smaller than threshold.
result = Pscore-Pscore;
for i = 1:length(Pscore)
    if best_threshold < Pscore(i)
        result(i) = 1;
    else
        result(i) = 0;
    end
end

end