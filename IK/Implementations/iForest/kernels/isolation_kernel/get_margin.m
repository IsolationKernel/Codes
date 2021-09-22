function [margin,pSV, nSV] = get_margin(model, y, ktrn)
   y(y==0) = -1;
   K = ktrn(model.sv_indices,model.sv_indices);
   c = sv_coef.*y(model.sv_indices);
   C = c*c';
   WW = K.*C;
   nm = sqrt(sum(WW(:)));
   margin = 1/nm;
   pSV = model.nSV(1);
   nSV = model.nSV(2);
end