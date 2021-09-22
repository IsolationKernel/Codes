function [ K ] = lap_kernel( x, y, T )
   K = zeros(size(x,1), size(y,1));
   %lambda = log2(T)/size(x,2);
   lambda = T;
   for i = 1:size(x,1)
       for j = 1:size(y,1)
           K(i,j) = -lambda*sum(abs(x(i,:) - y(j,:)));
       end
   end
   K = exp(K);
end

