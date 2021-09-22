function m_sim = KNN_AdaptiveGaussian(data, k)

m_dis=pdist2(data,data);
m_dis_sorted = sort(m_dis, 2);
sigq = m_dis_sorted(:, k);
clear m_dis_sorted
sigp=sigq';
m_sim = exp(-0.5*(m_dis.^2)./(sigq.*sigp)); 
%dis=1- m_sim;

%%
% N=size(data,1);
% J = knnsearch(data,data,'K',k);
% 
% I = repmat((1:N)', 1, size(J, 2));
% 
% A = accumarray([I(:) J(:)], 1);
% A = (A+A')/2;
% 
% dis=1-A;


end