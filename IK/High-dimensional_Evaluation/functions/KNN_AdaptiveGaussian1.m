function dis = KNN_AdaptiveGaussian1(data, k, p)
% Distance based on Adaptive Gaussian 

m_dis=pdist2(data,data,'minkowski',p);
m_dis_sorted = sort(m_dis, 2);
sigq = m_dis_sorted(:, k + 1);
clear m_dis_sorted
sigp=sigq';
m_sim = exp(-0.5*(m_dis.^2)./(sigq.*sigp)); 
dis=1- m_sim;
 

end