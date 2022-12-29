function[ best_Pscore, best_ent, best_psi ] = best_psi(Y, window)

% select the best psi by using approximate Entropy 

Psi_list = [2, 4, 8, 16, 32, 64];  %range of psi

Pscore = {};
ent = double(Psi_list - Psi_list);
parfor i = 1:length(Psi_list)
    Pscore{i} = point_score(Y, Psi_list(i), window);
    ent(i) = approximateEntropy(Pscore{i}); 
    % Approximate entropy is a measure to quantify the amount of regularity 
    % and unpredictability of fluctuations over a time series. 
    
end

[best_ent, idx] = min(ent); 
best_Pscore = Pscore{idx(1)};
best_psi = Psi_list(idx(1));