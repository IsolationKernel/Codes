%% Hubness calculation for Figure5

clear
clc
d = 3;

%% hubness count using Euclidean distance

data = unifrnd(0,1,5000,10000);
data = data(:,1:d);
[n,~] = size(data);
count1 = zeros(n);

dis=pdist2(data,data);
dis=dis./max(max(dis));

%% Gaussian
sigma = 5;
Gadis = Gaussian1(data, sigma);
Gadis=Gadis./max(max(Gadis));

f1 = sort(Gadis,2);

for iIndex = 1:n
   for jIndex = 1:n
      if Gadis(iIndex,jIndex)<f1(iIndex,7)
          count1(jIndex) = count1(jIndex) + 1;
      end
   end
end

fre1 = tabulate(count1(:));
fre1 = fre1';
[~,m] = size(fre1);
a1 = fre1(1:2,2:m);
a1(2,:) = a1(2,:)./n;

plot1 = plot((a1(1,:)),(a1(2,:)),'MarkerSize',10,'Linewidth',2);
% plot1 = plot(log10(a1(1,:)),log10(a1(2,:)),'MarkerSize',10,'Linewidth',2);

clear fre1;
clear count1;
clear Gadis;
clear f1;


count2 = zeros(n);

%% IK
t=100;
psi=32;  
[ ~, proximity ] = aNNE (dis, psi, t);
dis_anne = 1 - proximity;

f2 = sort(dis_anne,2);

for iIndex = 1:n
   for jIndex = 1:n
      if dis_anne(iIndex,jIndex)<f2(iIndex,7)
          count2(jIndex) = count2(jIndex) + 1;
      end
   end
end

fre2 = tabulate(count2(:));
fre2 = fre2';
[~,m] = size(fre2);
a2 = fre2(1:2,2:m);
a2(2,:) = a2(2,:)./n;

clear fre2;
clear count2;
clear dis_anne;
clear f2;
clear proximity;

hold on;

plot2 = plot((a2(1,:)),(a2(2,:)),'MarkerSize',10,'Linewidth',2);
% plot2 = plot(log10(a2(1,:)),log10(a2(2,:)),'MarkerSize',10,'Linewidth',2);

set(plot1,'DisplayName','GK','Color','g');
set(plot2,'DisplayName','IK','Color','r'); 

plot1.Marker = '^';
plot2.Marker = '*';

legend('location','best','FontSize',24);
set(gca,'FontSize',20);

ylabel({'$$p({O_{5}})$$'},'interpreter','latex','FontSize',28);
xlabel({'$${O_{5}}$$'},'interpreter','latex','FontSize',28);

% ylabel({'$${log_{10}(p({O_{5}}))}$$'},'interpreter','latex','FontSize',26);
% xlabel({'$${log_{10}({O_{5}})}$$'},'interpreter','latex','FontSize',26);

hold off;
