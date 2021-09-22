function [pred, acc, fv] = mklpredict( tst_label, tst_data, model )
    Kt=mklkernel(tst_data,model.InfoKernel,model.Weight,model.options,model.sv,model.beta);
    fv=Kt*model.sv_coef+model.b;
    pred = sign(fv);
    tst_label(tst_label~=1) = -1;
    acc = mean(pred==tst_label)
end

