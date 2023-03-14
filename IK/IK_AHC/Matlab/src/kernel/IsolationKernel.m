function m_sim = IsolationKernel(data, psi, t, sq)

    m_dist = pdist2(data, data);
    n = size(m_dist, 1);
    m_sim = zeros(size(m_dist));

    parfor i = 1:t % parfor MAY NOT work with acceleration by single GPU
        subIndex = datasample(1:n, psi, 'Replace', false);
        [~, centerIdx] = min(m_dist(subIndex, :));
        m_sim = m_sim + (centerIdx' == centerIdx);
    end

    m_sim = m_sim / t;

    if sq == false
        m = tril(true(size(m_sim)), -1);
        m_sim = m_sim(m)';
    end

end
