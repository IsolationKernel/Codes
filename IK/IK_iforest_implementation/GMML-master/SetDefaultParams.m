function s = SetDefaultParams(s)
% This function sets the default parameters of the method

if (isfield(s, 'mu') == 0),
    s.mu = 1e-04; %regularization parameter
end

if (isfield(s, 'const_factor') == 0),
    s.const_factor = 40; %the coefficient of the number of constraints
end

if (isfield(s, 'thresh') == 0),
   s.thresh = 1e-9; %the threshold for the auto-regularization
end

if (isfield(s, 'tuning_num_fold') == 0),
   s.tuning_num_fold = 5; %the number of folds in the tuning phase
end
end