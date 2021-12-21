clear;

T_td = readtable('td_xNNE_10D_10000.csv');
data_td = T_td{:,1:10};
[n,d]=size(data_td);

sets = 100;

for psi = [2, 8]
  mdl_fn = sprintf('mdl_xNNE_10D_10000_psi_%d_t_100.csv', psi);
  T_mdl = readtable(mdl_fn);
  data_mdl = T_mdl{:,1:10};

  all_zero_flag = double(data_td~=0);

  % calculate one non zero
  instances_num = size(data_td,1);

  idx = zeros(instances_num, sets);
  sim = zeros(instances_num,sets);

  for i = 0:sets - 1
    subIndex = [i * psi + 1 : (i + 1) * psi];
    mdl = data_mdl(subIndex, :);

    distance_sample = data_td .^2 * ones(size(mdl')) + ones(size(data_td)) * (mdl') .^2 - 2 * data_td * mdl';
    distance_sample2sample = mdl .^2 * ones(size(mdl')) + ones(size(mdl)) * (mdl') .^2 - 2 * mdl * mdl';
    distance_sample2sample(distance_sample2sample < 1e-9) = 99999999;
    distance_sample2sample = min(distance_sample2sample,[],2);
    [min_value,min_index] = min(distance_sample');
    ball_r = distance_sample2sample(min_index);
    ball_r = reshape(ball_r,1,instances_num);
    similarity = zeros(1,instances_num);
    max_min_value = max(min_value);
    min_min_value = 0;
    min_value2 = (min_value - min_min_value) ./ (max_min_value - min_min_value);
    similarity = min_value2 * 0;
    similarity(ball_r >= min_value) = 1;
    idx(:,i + 1) = min_index;
    sim(:,i + 1) = similarity;
  end

% probably better ways to do this
  res_fn = sprintf('./res_iNNE_10D_10000_psi_%d_t_100.csv', psi);
  fileID = fopen(res_fn, 'w');

  for i = 1:n
    for j = 1:sets
      if sim(i,j) == 1
        mrk = idx(i,j);
        for k = 1:mrk - 1
          fprintf(fileID, '0,');
        end
        
        fprintf(fileID, '1,');

        for k = mrk + 1: psi
          fprintf(fileID, '0,');
        end
      else
        for k = 1:psi
          fprintf(fileID, '0,');
        end
      end
    end
    
    ch = char(T_td{i,11});
    fprintf(fileID, '%s\n', ch);
  end

  fclose(fileID);
end
