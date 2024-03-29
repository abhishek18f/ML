function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).



for q = 1:(size(z))(1);
  for w = 1: (size(z))(2);
     g(q,w) = 1/(1+exp(-z(q,w)));
  endfor
endfor




% =============================================================

end
