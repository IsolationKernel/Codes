function m_sim = KNN_AdaptiveGaussian(data, k)
% calculate the adaptive gaussian similarity 

m_dis=pdist2(data,data);
m_dis_sorted = sort(m_dis, 2);
sigq = m_dis_sorted(:, k);
clear m_dis_sorted
sigp=sigq';
m_sim = exp(-0.5*(m_dis.^2)./(sigq.*sigp));  


end