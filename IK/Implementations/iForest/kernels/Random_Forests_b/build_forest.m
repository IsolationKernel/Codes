function [ model ] = build_forest(label, feature, psi )
model = train_RF(feature, label,'ntrees', 100,'oobe','n','nsamtosample',psi,'method','g','nvartosample', size(feature,2));
%model = train_RF(feature, label,'ntrees', 100,'oobe','y','nsamtosample',psi,'method','g','nvartosample',2);

end

