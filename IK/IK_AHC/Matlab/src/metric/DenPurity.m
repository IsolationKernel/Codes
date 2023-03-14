function [Purity] = DenPurity(L, class)
    %class: true label
    %% building parent index matrix
    %%%%  row is instance index collunm is parent index
    n = size(L, 1) + 1; % number of instances
    Tclass = [1:length(class)]';

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

    C = unique(class);
    T = unique(Tclass);
    Pure = 0;
    S = 0;
    tc_index = zeros(length(C), 2 * n - 1);
    subtree_sum = zeros(1, n);

    for ci = 1:length(C)

        for ti = 1:n
            subtree = find(Tclass == T(ti));
            subtree_sum(ti) = length(subtree);
            tc = find(class(subtree) == C(ci));
            tc_index(ci, ti) = length(tc) / length(subtree);
        end

        for ti = n + 1:2 * n - 1
            Tinstances = find(ParentMatrix(:, ti - n) == 1);
            p = 0;

            for i = 1:length(Tinstances)
                p = p + tc_index(ci, Tinstances(i)) * subtree_sum(Tinstances(i));
            end

            tc_index(ci, ti) = p / sum(subtree_sum(Tinstances));
        end

    end

    for Ci = 1:length(C)
        CurrentInstance = find(class == C(Ci));

        for i = 1:length(CurrentInstance)

            for j = i + 1:length(CurrentInstance)
                tc1 = Tclass(CurrentInstance(i));
                tc2 = Tclass(CurrentInstance(j));

                if tc1 == tc2 % The lca is the pskc cluster
                    Pure = Pure + tc_index(Ci, tc1);

                else
                    ShareParent = find(ParentMatrix(tc1, :) .* ParentMatrix(tc2, :) == 1); % find the nearest parent

                    Pure = Pure + tc_index(Ci, ShareParent(1) + n);
                end

                S = S + 1;
            end

        end

    end

    Purity = Pure / S;
end
