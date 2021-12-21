function [ forest ] = iForestTrain(data, psi, num_trees, max_height, num_features)
%    Each row in data is an instance.
%    Psi is the size of each subset.     
%
%%
   d_psi = 256;
   d_num_trees = 100;
   d_max_height = 8;
   d_num_features = size(data,2);
   switch(nargin)
       case 1 
       case 2 
           d_psi = psi;
           d_max_height = round(log2(psi));
       case 3
           d_psi = psi;
           d_max_height = round(log2(psi));
           d_num_trees = num_trees;
       case 4
           disp('ERROR: Inputs do not match!\n');
       case 5
           d_psi = psi;
           d_num_trees = num_trees;
           d_max_height = max_height;
           d_num_features = num_features;
   end
   forest = build_iForest(data, d_psi, d_num_trees, d_max_height, d_num_features);
end

