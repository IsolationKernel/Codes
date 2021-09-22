function [DP_F,AGDP_F,IDP_F,SC_F,AGSC_F,ISC_F,DP_A,AGDP_A,IDP_A,SC_A,AGSC_A,ISC_A] = clustering(data,class)



%% DP

% distance
DisMatrix=pdist2(data,data);

[DP_F,DP_A]=testDP(data,class,DisMatrix);

% Isolation kernel - aNNE
t=200;
F2=[];
A2=[];
for i=1:10
    psi=2^i;
    if psi>size(data,1)
        break
    end
    iF2=[];
    iA2=[];
    for j=1:10
        [iF2(j),iA2(j)]=testDP(data,class,1-aNNE_similarity(DisMatrix, psi, t));
    end
    F2(i)=mean(iF2);
    A2(i)=mean(iA2);
end

[IDP_F]=max(F2);
[IDP_A]=max(A2);


% adaptive Gaussian kernel
F3=[];
A3=[];
for i=1:10
    kn=ceil(size(data,1)*0.05*i);
    AG=KNN_AdaptiveGaussian(data, kn);
    AG(isnan(AG))=0;
    [F3(i),A3(i)]=testDP(data,class,1-AG);
end
[AGDP_F]=max(F3);
[AGDP_A]=max(A3);

%% SC


% Gaussian kernel
%B=[2.^[-5:1:5] 2.^[-5:1:5]*size(data,2)];
B=2.^[-5:1:5]*size(data,2);
F=[];
A=[];
k=length(unique(class));

DisMatrix=pdist2(data,data);
for i=1:10
    sigma=B(i);
    S = exp(-(DisMatrix.^2 / (2*sigma^2)));
    iF=[];
    iA=[];
    for j=1:10
        Tclass = sc(S, k);
        iF(j)= fmeasure(class,Tclass);
        iA(j)=ami(class',Tclass');
    end
    F(i)=mean(iF);
    A(i)=mean(iA);
end
SC_F=max(F);
SC_A=max(A);

% Isolation kernel - aNNE
t=200;
F2=[];
A2=[];
for i=1:10
    psi=2^i;
    if psi>size(data,1)
        break
    end
    iF=[];
    iA=[];
    for j=1:10
        Tclass = sc(aNNE_similarity(DisMatrix, psi, t), k);
        iF(j)= fmeasure(class,Tclass);
        iA(j)=ami(class',Tclass');
    end
    F2(i)=mean(iF);
    A2(i)=mean(iA);
end
ISC_F=max(F2);
ISC_A=max(A2);

% adaptive Gaussian kernel

F3=[];
A3=[];
for i=1:10
    kn=ceil(size(data,1)*0.05*i);
    AG=KNN_AdaptiveGaussian(data, kn);
    AG(isnan(AG))=0;
    iF=[];
    iA=[];
    for j=1:10
        Tclass = sc(AG, k);
        iF(j)= fmeasure(class,Tclass);
        iA(j)=ami(class',Tclass');
    end
    F3(i)=mean(iF);
    A3(i)=mean(iA);
end
AGSC_F=max(F3);
AGSC_A=max(A3);

end

