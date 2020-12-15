function K=mklkernel(xapp,InfoKernel,Weight,options,xsup,beta)

if nargin <5
    xsup=xapp;
    beta=[];


    for k=1:length(Weight)

        Kr=svmkernel(xapp(:,InfoKernel(k).variable),InfoKernel(k).kernel,InfoKernel(k).kerneloption, xsup(:,InfoKernel(k).variable));

        Kr=Kr*Weight(k);
%         if options.efficientkernel
%             Kr=build_efficientK(Kr);
%         end;

        K(:,:,k)=Kr;


    end;
else
    ind=find(beta);
    K=zeros(size(xapp,1),size(xsup,1));
    for i=1:length(ind);
        k=ind(i); 
        Kr=svmkernel(xapp(:,InfoKernel(k).variable),InfoKernel(k).kernel,InfoKernel(k).kerneloption, xsup(:,InfoKernel(k).variable));
        Kr=Kr*Weight(k);
        K=K+ Kr*beta(k);
    end;

end;