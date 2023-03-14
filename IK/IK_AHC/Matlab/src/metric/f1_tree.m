function f1_score = f1_tree(Z, class, class_num)

    f1_score = 0;

    for cl = class_num
        idx = cluster(Z, 'maxclust', cl);
        f1 = Fmeasure(class, idx);
        f1_score = max(f1_score, f1);
    end

end
