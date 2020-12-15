function all_im = convert_point( data,voros , t, psi)
%CONVERT_TO_IM return all isoaltion set by voros
%% 
num_bag = size(data,1);
voro_num = size(voros,3);
all_im = zeros(num_bag, psi*voro_num);
for i=1:num_bag
    im = zeros(t,psi);
    index = voros(i,1,:);
    index = reshape(index,1,t);
    index_t = 1:t;
    linear_index = sub2ind(size(im),index_t,index);
    similarity = voros(i,3,:);
    im(linear_index) = similarity;
    im = reshape(im,1,psi*voro_num);
    all_im(i,:)=im;
end
end

