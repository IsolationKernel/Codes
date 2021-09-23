function [ m_similarity ] = aNNE_similarity (m_distance, psi, t)
% Nearest-neighbour-induced isolation similarity (Isolation Kernel based on voronoi diagram partitioning)
% Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C., 2019, July. Nearest-neighbour-induced isolation similarity and its impact on density-based clustering.

n = size(m_distance, 1);
m_similarity = m_distance - m_distance;

parfor i = 1:t % parfor MAY NOT work with acceleration by single GPU
    subIndex = datasample(1:n, psi, 'Replace', false);        
    [~, centerIdx] = min(m_distance(subIndex, :));    
    m_similarity = m_similarity + (centerIdx' == centerIdx);
end
m_similarity = m_similarity/t;

end