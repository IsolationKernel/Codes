function dis = Gaussian1(data, sigma)

m_dis=pdist2(data,data);
m_sim = exp(-0.5*(m_dis.^2)./(sigma*sigma)); 
dis=1- m_sim;
