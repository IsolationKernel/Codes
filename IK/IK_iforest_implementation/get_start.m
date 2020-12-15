currPath = fileparts(mfilename('fullpath'));% get current path

cd('svm/matlab/');
make();
cd(currPath);

cd('kernels/isolation_kernel/');
mex build_iForest.cpp 
mex get_mer.cpp
cd(currPath);

cd('kernels/Random_Forests/cartree/mx_files/');
mx_compile_cartree
cd(currPath);

cd('kernels/Random_Forests_b/cartree/mx_files/');
mx_compile_cartree
cd(currPath);

cd('simplemkl/');
mex devectorize.c
mex vectorize.c
cd(currPath);