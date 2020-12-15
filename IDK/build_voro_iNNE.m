function voros = build_voros( train_data_all, test_data_all, psi, voro_num, k)
% split distance, only calculate sample index

%non-tree based model, deal with zero value
all_instances = [train_data_all; test_data_all];
all_zero_flag = double(all_instances~=0);
train_zero_flag = double(train_data_all~=0);

% calculate one non zero
instances_num = size(all_instances,1);

voros = zeros(instances_num,3,voro_num);
for i=1:voro_num
    sample_index = randperm(size(train_data_all,1));
    sample_index = sample_index(1:psi);
    % only calculate sample index
    train_data_all_sample = train_data_all(sample_index,:);
    distance_sample = all_instances.^2*ones(size(train_data_all_sample'))+ones(size(all_instances))*(train_data_all_sample').^2-2*all_instances*train_data_all_sample';
    distance_sample2sample = train_data_all_sample.^2*ones(size(train_data_all_sample'))+ones(size(train_data_all_sample))*(train_data_all_sample').^2-2*train_data_all_sample*train_data_all_sample';
    distance_sample2sample(distance_sample2sample<1e-9) = 99999999;
    distance_sample2sample = min(distance_sample2sample,[],2);
    [min_value,min_index] = min(distance_sample');
    ball_r = distance_sample2sample(min_index);
    ball_r = reshape(ball_r,1,instances_num);
    similarity = zeros(1,instances_num);
%     similarity(ball_r<min_value) = (1+min_value(ball_r<min_value).^2).^(-1);
    %similarity = (1+(min_value*10).^2).^(-1);
    max_min_value = max(min_value);
    min_min_value = 0;
    min_value2 = (min_value-min_min_value)./(max_min_value-min_min_value);
    %similarity = (1+(min_value2*20).^2).^(-1);
    similarity = min_value2*0;
    similarity(ball_r>=min_value)=1;
    voros(:,1,i)=min_index;
    voros(:,2,i)=min_value;
    voros(:,3,i)=similarity;
end
    
end

