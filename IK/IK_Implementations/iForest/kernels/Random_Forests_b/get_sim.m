function [ K ] = get_sim( insta, instb, model )

na = size(insta,1);
nb = size(instb,1);
K = zeros(na, nb);
heights = zeros(100,1);
for i = 1:100
    mxheight = max(model(i).nodeheight);
    heights(i) = mxheight;
end
[~, lv1]  = eval_RF(insta, model, heights, 'oobe', 'n');
[~, lv2]  = eval_RF(instb, model,  heights, 'oobe', 'n');
for i = 1: na
    for j = 1:nb
        K(i,j) = sum(lv1(:,i)==lv2(:,j))/100;
    end
end
end

