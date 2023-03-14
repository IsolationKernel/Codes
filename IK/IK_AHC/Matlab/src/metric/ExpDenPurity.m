function [Purity] = ExpDenPurity(L, class)

    %% building parent index matrix
    %%%%  row is instance index collunm is parent index
    n = size(L, 1) + 1; % number of instances

    L(:, end + 1) = (size(L, 1) + (1:size(L, 1))) + 1;

    ParentMatrix = zeros(n, 2 * n);

    for i = 1:size(L, 1)
        CurrentIndex = [];

        if L(i, 1) > n
            ind = find(ParentMatrix(:, L(i, 1)) == 1)';
            CurrentIndex = [CurrentIndex ind];
        else
            CurrentIndex = [CurrentIndex L(i, 1)];
        end

        if L(i, 2) > n
            ind = find(ParentMatrix(:, L(i, 2)) == 1)';
            CurrentIndex = [CurrentIndex ind];
        else
            CurrentIndex = [CurrentIndex L(i, 2)];
        end

        ParentMatrix(CurrentIndex, L(i, 4)) = 1;
    end

    ParentMatrix(:, [1:n 2 * n]) = [];

    %% scanning

    C = unique(class);
    Pure = 0;
    S = 0;

    for Ci = 1:length(C)
        CurrentInstance = find(class == C(Ci));
        n = floor(0.5 * length(CurrentInstance));

        for sam = 1:n
            sample_i = randsample(CurrentInstance, 1, 'false');
            sample_j = randsample(CurrentInstance, 1, 'false');
            ShareParent = find(ParentMatrix(sample_i, :) .* ParentMatrix(sample_j, :) == 1); % find the nearest parent
            instances = find(ParentMatrix(:, ShareParent(1)) == 1); % instances sharing the same parent
            Pure = Pure + sum(class(instances) == C(Ci)) / length(instances); % purity score
            S = S + 1;
        end

    end

    Purity = Pure / S; % average purity score
end
