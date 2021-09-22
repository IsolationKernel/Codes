clear;

T_td = readtable('td_xNNE_10D_10000.csv');
data_td = T_td{:,1:10};
[n,d]=size(data_td);

sets = 100;

for psi = [2, 8]
  mdl_fn = sprintf('mdl_xNNE_10D_10000_psi_%d_t_100.csv', psi);
  T_mdl = readtable(mdl_fn);
  data_mdl = T_mdl{:,1:10};

  c=0:psi:(n-1)*psi;

  ndata=[];

  for i = 0:sets - 1
    subIndex = [i * psi + 1 : (i + 1) * psi];
    mdl = data_mdl(subIndex, :);
    dist = pdist2(mdl, data_td);
    [~, centerIdx] = min(dist);
    z=zeros(psi,n);
    z(centerIdx+c)=1;
    ndata=[ndata z'];
  end

  T_res = array2table(ndata);
  T_res.class =  T_td{:,11};

  res_fn = sprintf('./res_aNNE_10D_10000_psi_%d_t_100.csv', psi);
  writetable(T_res, res_fn, 'WriteVariableNames', false);
end
